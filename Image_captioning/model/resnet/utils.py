import os
import numpy as np
import h5py
import json
import torch
from PIL import Image          # thay thế scipy.misc.imread / imresize (bị xóa từ scipy 1.3)
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


# ── Image helpers (PIL thay cho scipy.misc) ────────────────────────────────────
def imread(path):
    """Đọc ảnh → numpy array HWC uint8."""
    return np.array(Image.open(path))


def imresize(img, size):
    """
    Resize numpy array HWC về (height, width) cho trước.
    PIL nhận size theo thứ tự (width, height), hàm này nhận (height, width)
    để tương thích với cách dùng cũ: imresize(img, (256, 256)).
    """
    h, w = size
    return np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR))


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image,
                       min_word_freq, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    train_image_paths    = []
    train_image_captions = []
    val_image_paths      = []
    val_image_captions   = []
    test_image_paths     = []
    test_image_captions  = []
    word_freq            = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = (
            os.path.join(image_folder, img['filepath'], img['filename'])
            if dataset == 'coco'
            else os.path.join(image_folder, img['filename'])
        )

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths)   == len(val_image_captions)
    assert len(test_image_paths)  == len(test_image_captions)

    words    = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>']   = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>']   = len(word_map) + 1
    word_map['<pad>']   = 0

    base_filename = (
        dataset + '_' + str(captions_per_image) + '_cap_per_img_'
        + str(min_word_freq) + '_min_word_freq'
    )

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)
    for impaths, imcaps, split in [
        (train_image_paths, train_image_captions, 'TRAIN'),
        (val_image_paths,   val_image_captions,   'VAL'),
        (test_image_paths,  test_image_captions,  'TEST'),
    ]:
        with h5py.File(
            os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a'
        ) as h:
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens      = []

            for i, path in enumerate(tqdm(impaths)):
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [
                        choice(imcaps[i])
                        for _ in range(captions_per_image - len(imcaps[i]))
                    ]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                assert len(captions) == captions_per_image

                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                for j, c in enumerate(captions):
                    enc_c = (
                        [word_map['<start>']]
                        + [word_map.get(word, word_map['<unk>']) for word in c]
                        + [word_map['<end>']]
                        + [word_map['<pad>']] * (max_len - len(c))
                    )
                    c_len = len(c) + 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            with open(
                os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w'
            ) as j:
                json.dump(enc_captions, j)

            with open(
                os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w'
            ) as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """Fills embedding tensor with values from the uniform distribution."""
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    print("\nLoading embeddings...")
    for line in tqdm(open(emb_file, 'r')):
        line      = line.split(' ')
        emb_word  = line[0]
        embedding = list(map(
            lambda t: float(t),
            filter(lambda n: n and not n.isspace(), line[1:])
        ))
        if emb_word not in vocab:
            continue
        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """Clips gradients computed during backpropagation to avoid explosion of gradients."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, bleu4, is_best):
    """Saves model checkpoint."""
    state = {
        'epoch':                    epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4':                   bleu4,
        'encoder':                  encoder,
        'decoder':                  decoder,
        'encoder_optimizer':        encoder_optimizer,
        'decoder_optimizer':        decoder_optimizer,
    }
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """Keeps track of most recent, average, sum, and count of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """Shrinks learning rate by a specified factor."""
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind     = scores.topk(k, 1, True, True)
    correct    = ind.eq(targets.view(-1, 1).expand_as(ind))
    return correct.view(-1).float().sum().item() * (100.0 / batch_size)
