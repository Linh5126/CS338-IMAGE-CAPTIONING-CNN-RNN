import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import (
    AverageMeter, adjust_learning_rate, accuracy,
    save_checkpoint, clip_gradient,
)
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
import os
import json

nltk.download('wordnet', quiet=True)

# Data parameters
data_folder = '/content/drive/MyDrive/CS338'
data_name   = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim       = 512
attention_dim = 512
decoder_dim   = 512
dropout       = 0.5
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch              = 0
epochs                   = 120
epochs_since_improvement = 0
batch_size               = 32
workers                  = 1
encoder_lr               = 1e-4
decoder_lr               = 4e-4
grad_clip                = 5.
alpha_c                  = 1.
best_bleu4               = 0.
print_freq               = 100
fine_tune_encoder        = False
checkpoint               = None


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch
    global fine_tune_encoder, data_name, word_map

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            dropout=dropout,
        )
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder else None
        )
    else:
        checkpoint_data          = torch.load(checkpoint, weights_only=False)
        start_epoch              = checkpoint_data['epoch'] + 1
        epochs_since_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4               = checkpoint_data['bleu-4']
        decoder                  = checkpoint_data['decoder']
        decoder_optimizer        = checkpoint_data['decoder_optimizer']
        encoder                  = checkpoint_data['encoder']
        encoder_optimizer        = checkpoint_data['encoder_optimizer']
        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # FIX: pin_memory chỉ True khi dùng CUDA (tránh warning + RAM tốn thêm trên CPU)
    _pin = device.type == 'cuda'

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=_pin,
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=_pin,
    )

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder and encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader, encoder, decoder, criterion,
              encoder_optimizer, decoder_optimizer, epoch)

        recent_bleu4 = validate(val_loader, encoder, decoder, criterion, word_map)

        is_best    = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0

        save_checkpoint(data_name, epoch, epochs_since_improvement,
                        encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion,
          encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top5accs   = AverageMeter()
    start      = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs    = imgs.to(device)
        caps    = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss  = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')


def validate(val_loader, encoder, decoder, criterion, word_map):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses     = AverageMeter()
    top5accs   = AverageMeter()
    start      = time.time()

    references     = []
    hypotheses     = []
    references_str = []
    hypotheses_str = []

    rev_word_map = {v: k for k, v in word_map.items()}
    rouge_sc     = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # FIX: thêm <end> vào skip — trước đây <end> còn trong reference
    # khiến BLEU/METEOR/ROUGE tính thấp hơn thực tế vì hypothesis không có <end>
    skip = {word_map['<start>'], word_map['<end>'], word_map['<pad>']}

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs    = imgs.to(device)
            caps    = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )
            targets     = caps_sorted[:, 1:]
            scores_copy = scores.clone()

            scores  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss  = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(f'Validation: [{i}/{len(val_loader)}]\t'
                      f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')

            # ── References ─────────────────────────────────────────────────────
            # FIX: sort_ind nằm trên GPU, allcaps nằm trên CPU → .cpu() trước khi index
            allcaps = allcaps[sort_ind.cpu()]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions_tok = [
                    [w for w in c if w not in skip]
                    for c in img_caps
                ]
                references.append(img_captions_tok)
                references_str.append([
                    " ".join(rev_word_map.get(w, "") for w in cap)
                    for cap in img_captions_tok
                ])

            # ── Hypotheses ─────────────────────────────────────────────────────
            _, preds = torch.max(scores_copy, dim=2)
            preds = [preds[j][:decode_lengths[j]].tolist() for j in range(len(preds))]
            hypotheses.extend(preds)
            for p in preds:
                hypotheses_str.append(
                    " ".join(rev_word_map.get(w, "") for w in p)
                )

            assert len(references) == len(hypotheses)

        # ── BLEU ──────────────────────────────────────────────────────────────
        bleu1 = corpus_bleu(references, hypotheses, weights=(1,    0,    0,    0   ))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,  0.5,  0,    0   ))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0   ))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        # ── METEOR ────────────────────────────────────────────────────────────
        meteor_scores = [
            meteor_score([r.split() for r in refs], hyp.split())
            for refs, hyp in zip(references_str, hypotheses_str)
        ]
        meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        # ── ROUGE-L ───────────────────────────────────────────────────────────
        rouge_scores = [
            max(rouge_sc.score(r, hyp)['rougeL'].fmeasure for r in refs)
            for refs, hyp in zip(references_str, hypotheses_str)
        ]
        rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        print(f'\n* VAL LOSS: {losses.avg:.3f} | TOP-5 ACC: {top5accs.avg:.3f}')
        print(f'* BLEU-1: {bleu1:.4f} | BLEU-2: {bleu2:.4f} | '
              f'BLEU-3: {bleu3:.4f} | BLEU-4: {bleu4:.4f}')
        print(f'* METEOR: {meteor:.4f} | ROUGE-L: {rouge_l:.4f}\n')

    return bleu4


if __name__ == '__main__':
    main()
