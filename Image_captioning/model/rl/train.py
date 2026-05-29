import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import GradScaler, autocast   # AMP — tăng tốc ~1.5-2x trên GPU
from models import DualStreamEncoder, DecoderAdaptive
from datasets import CaptionDataset
from pycocoevalcap.cider.cider import Cider
from utils import (
    AverageMeter,
    adjust_learning_rate,
    accuracy,
    save_checkpoint,
    clip_gradient,
)
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import nltk
import argparse
import os
import json

nltk.download('wordnet', quiet=True)
from nltk.translate.meteor_score import meteor_score


# ── Argument Parser ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Image Captioning — DualStream Encoder + Adaptive LSTM Decoder'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='flickr8k',
    choices=['flickr8k', 'flickr30k'],    # FIX: 'flickr32k' → 'flickr30k'
    help='Dataset to train on',
)
parser.add_argument('--encoder', default='dual_stream')
parser.add_argument('--decoder', default='lstm_adaptive')

# parse_known_args() tránh crash trong Jupyter/Kaggle do kernel argv
args, _ = parser.parse_known_args()


# ── Paths ──────────────────────────────────────────────────────────────────────
data_folder     = '/kaggle/input/datasets/llnhins/cs338l/'
data_name       = f'{args.dataset}_5_cap_per_img_5_min_word_freq'
checkpoint_name = f'{args.encoder}_{args.decoder}_{data_name}'


# ── Hyper-parameters ───────────────────────────────────────────────────────────
emb_dim       = 512
attention_dim = 512
decoder_dim   = 512
dropout       = 0.5
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# AMP: chỉ bật khi có CUDA (GradScaler không hỗ trợ CPU)
use_amp = device.type == 'cuda'

start_epoch              = 0
epochs                   = 120
epochs_since_improvement = 0
batch_size               = 32
workers                  = 4       # FIX: 1 → 4 để prefetch ảnh song song
encoder_lr               = 1e-5
decoder_lr               = 5e-5
grad_clip                = 5.
alpha_c                  = 1.
best_bleu4               = 0.
print_freq               = 100
fine_tune_encoder        = True
checkpoint               = '/kaggle/input/datasets/llnhins/cs338l/BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch
    global fine_tune_encoder, data_name

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # ── Khởi tạo hoặc load checkpoint ─────────────────────────────────────────
    if checkpoint is None:
        encoder = DualStreamEncoder(encoded_image_size=7)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder else None
        )
        decoder = DecoderAdaptive(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            encoder_dim=2048,
            dropout=dropout,
        )
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
    else:
        checkpoint_data          = torch.load(checkpoint, weights_only=False)
        start_epoch              = checkpoint_data['epoch'] + 1
        
        # 1. Reset lại bộ đếm kiên nhẫn vì RL là một chặng đường mới
        epochs_since_improvement = 0 
        best_bleu4               = checkpoint_data['bleu-4']
        
        # 2. Chỉ load não (Weights), KHÔNG LOAD OPTIMIZER CŨ
        decoder                  = checkpoint_data['decoder']
        encoder                  = checkpoint_data['encoder']

        # 3. Khởi tạo Optimizer mới toanh với Learning Rate RL siêu nhỏ (5e-5)
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )

        if fine_tune_encoder:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
        else:
            encoder_optimizer = None

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # AMP scaler — no-op khi use_amp=False
    scaler = GradScaler(enabled=use_amp)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # FIX: pin_memory chỉ True khi dùng CUDA (tránh tốn RAM & warning trên CPU)
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

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            print("No improvement for 20 epochs — stopping early.")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder and encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader, encoder, decoder,
              encoder_optimizer, decoder_optimizer, epoch, word_map)

        recent_bleu4, recent_meteor, recent_rouge_l = validate(
            val_loader, encoder, decoder, criterion, word_map
        )

        is_best    = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print(f"Epochs since improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0

        # FIX: lưu cả METEOR và ROUGE-L vào checkpoint
        save_checkpoint(
            checkpoint_name, epoch, epochs_since_improvement,
            encoder, decoder, encoder_optimizer, decoder_optimizer,
            recent_bleu4, is_best,
            meteor=recent_meteor,
            rouge_l=recent_rouge_l,
        )


# ── Train một epoch ────────────────────────────────────────────────────────────
def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, word_map):
    decoder.train()
    if encoder is not None:
        encoder.train()

    batch_time = AverageMeter()
    reward_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    cider_scorer = Cider()
    rev_word_map = {v: k for k, v in word_map.items()}
    
    start_token = word_map['<start>']
    end_token = word_map['<end>']
    skip_tokens = {word_map['<start>'], word_map['<pad>'], word_map['<end>']}

    for i, (imgs, caps, caplens, allcaps) in enumerate(train_loader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # ── 1. Extract Features ────────────────────────────────────────────────
        if encoder is not None:
            imgs, padding_mask = encoder(imgs)
            padding_mask = padding_mask.to(device)
        else:
            padding_mask = None

        # ── 2. SCST Sampling & Greedy ──────────────────────────────────────────
        decoder.eval() # Greedy decode không cần track gradient
        with torch.no_grad():
            greedy_ids, _ = decoder.forward_greedy(
                imgs, start_token, end_token, max_len=50, padding_mask=padding_mask
            )
        
        decoder.train() # Bật lại train mode cho sample để tính gradient
        sample_ids, log_probs, sample_lengths = decoder.forward_sample(
            imgs, start_token, end_token, max_len=50, padding_mask=padding_mask
        )

        # ── 3. Chuẩn bị định dạng cho pycocoevalcap ───────────────────────────
        reward_start = time.time()
        
        gts = {}
        res_sample = {}
        res_greedy = {}

        for j in range(batch_size):
            # Ground Truths
            img_captions = [
                " ".join([rev_word_map.get(w, "") for w in c.tolist() if w not in skip_tokens])
                for c in allcaps[j]
            ]
            gts[j] = img_captions

            # Sampled Hypothesis
            s_words = [w for w in sample_ids[j].tolist() if w not in skip_tokens]
            res_sample[j] = [" ".join([rev_word_map.get(w, "") for w in s_words])]

            # Greedy Hypothesis
            g_words = [w for w in greedy_ids[j].tolist() if w not in skip_tokens]
            res_greedy[j] = [" ".join([rev_word_map.get(w, "") for w in g_words])]

        # ── 4. Tính CIDEr Reward ───────────────────────────────────────────────
        _, cider_sample = cider_scorer.compute_score(gts, res_sample)
        _, cider_greedy = cider_scorer.compute_score(gts, res_greedy)

        reward_time.update(time.time() - reward_start)

        # ── 5. Tính Loss & Update ──────────────────────────────────────────────
        reward = cider_sample - cider_greedy                         # [B] numpy array
        reward = torch.tensor(reward, dtype=torch.float32, device=device)

        # Tính tổng log_probs của câu sample (mask những timestep sau <end>)
        max_len = log_probs.size(1)
        sl_tensor = sample_lengths.unsqueeze(1)                      # [B, 1]
        t_range = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
        mask = (t_range < sl_tensor).float()                         # [B, max_len]

        masked_log_probs = log_probs * mask                          # [B, max_len]
        seq_log_probs = masked_log_probs.sum(dim=1)                  # [B]

        # Áp dụng công thức Policy Gradient
        loss = - (reward * seq_log_probs).mean()

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

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f}\t'
                  f'Reward Calc {reward_time.val:.3f}\t'
                  f'Mean Reward {reward.mean().item():.4f}\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')


# ── Validate ───────────────────────────────────────────────────────────────────
def validate(val_loader, encoder, decoder, criterion, word_map):
    """
    Returns: (bleu4, meteor, rouge_l)
    """
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
    scorer       = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Token ids cần loại khỏi references
    # FIX: thêm <end> — trước đây chỉ lọc <start> và <pad>, khiến BLEU tính sai
    skip_tokens = {word_map['<start>'], word_map['<pad>'], word_map['<end>']}

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs    = imgs.to(device)
            caps    = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

            with autocast(enabled=use_amp):
                if encoder is not None:
                    imgs, padding_mask = encoder(imgs)
                    padding_mask = padding_mask.to(device)
                else:
                    padding_mask = None

                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                    imgs, caps, caplens, padding_mask
                )
                targets     = caps_sorted[:, 1:]
                scores_copy = scores.clone()

                scores_packed  = pack_padded_sequence(
                    scores,  decode_lengths, batch_first=True
                ).data
                targets_packed = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                ).data
                loss = criterion(scores_packed, targets_packed)

                max_T     = alphas.size(1)
                dl_tensor = torch.tensor(decode_lengths, dtype=torch.float, device=device)
                t_idx     = torch.arange(max_T, device=device).unsqueeze(0).float()
                time_mask = (t_idx < dl_tensor.unsqueeze(1)).float()
                alpha_sum = (alphas * time_mask.unsqueeze(2)).sum(dim=1)
                loss += alpha_c * ((1.0 - alpha_sum) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores_packed, targets_packed, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            # ── References ─────────────────────────────────────────────────────
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                # FIX: lọc cả <end> — trước đây <end> còn trong reference
                #      làm BLEU/METEOR/ROUGE đo thấp hơn thực tế
                img_captions = [
                    [w for w in c if w not in skip_tokens]
                    for c in img_caps
                ]
                references.append(img_captions)
                ref_strs = [
                    " ".join(rev_word_map.get(w, "") for w in cap)
                    for cap in img_captions
                ]
                references_str.append(ref_strs)

            # ── Hypotheses ─────────────────────────────────────────────────────
            _, preds = torch.max(scores_copy, dim=2)
            preds_list = [
                [w for w in preds[j][: decode_lengths[j]].tolist() if w not in skip_tokens]
                for j in range(len(preds))
            ]
            hypotheses.extend(preds_list)
            for p in preds_list:
                hypotheses_str.append(
                    " ".join(rev_word_map.get(w, "") for w in p)
                )

        # ── BLEU ──────────────────────────────────────────────────────────────
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0,    0,    0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5,  0,    0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25, 0.25, 0.25))

        # ── METEOR ────────────────────────────────────────────────────────────
        meteor_scores = []
        for ref_list, hyp in zip(references_str, hypotheses_str):
            ref_tok = [r.split() for r in ref_list]
            hyp_tok = hyp.split()
            meteor_scores.append(meteor_score(ref_tok, hyp_tok))
        meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        # ── ROUGE-L ───────────────────────────────────────────────────────────
        rouge_scores = []
        for ref_list, hyp in zip(references_str, hypotheses_str):
            best_r = max(
                scorer.score(r, hyp)['rougeL'].fmeasure for r in ref_list
            )
            rouge_scores.append(best_r)
        rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        print(
            f'\n{"─"*60}\n'
            f' VAL Loss {losses.avg:.3f} | Top-5 Acc {top5accs.avg:.3f}\n'
            f' BLEU-1 {bleu1:.4f} | BLEU-2 {bleu2:.4f} | '
            f'BLEU-3 {bleu3:.4f} | BLEU-4 {bleu4:.4f}\n'
            f' METEOR {meteor:.4f} | ROUGE-L {rouge_l:.4f}\n'
            f'{"─"*60}\n'
        )

    return bleu4, meteor, rouge_l


if __name__ == '__main__':
    main()
