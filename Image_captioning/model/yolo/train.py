"""
train_baseline.py — Training script cho phiên bản BASELINE.

Giống hệt train.py, chỉ khác:
  1. Import DecoderBaseline từ models_baseline (không có sentinel)
  2. checkpoint_name có hậu tố 'lstm_baseline' để không ghi đè checkpoint chính
  3. alpha_c regularization dùng alphas [B, T, N] thay vì [B, T, N+1]
     (nhưng công thức vẫn y chang — chỉ số chiều cuối tự khớp)

Mọi hyperparameter khác (lr, batch_size, grad_clip, alpha_c, early stopping...)
đều giống train.py để đảm bảo so sánh công bằng.
"""

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

# ── Điểm khác biệt duy nhất so với train.py ──────────────────────────────────
from models import DualStreamEncoder, DecoderBaseline
# ─────────────────────────────────────────────────────────────────────────────

from datasets import CaptionDataset
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
    description='Image Captioning — DualStream Encoder + Baseline LSTM Decoder'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='flickr8k',
    choices=['flickr8k', 'flickr30k'],
)
parser.add_argument('--encoder', default='dual_stream')
parser.add_argument('--decoder', default='lstm_baseline')   # khác với train.py

args, _ = parser.parse_known_args()


# ── Paths ──────────────────────────────────────────────────────────────────────
data_folder     = '/kaggle/input/datasets/llnhins/cs338l/'
data_name       = f'{args.dataset}_5_cap_per_img_5_min_word_freq'
checkpoint_name = f'{args.encoder}_{args.decoder}_{data_name}'
# → ví dụ: 'dual_stream_lstm_baseline_flickr8k_5_cap_per_img_5_min_word_freq'
# Khác với train.py ('dual_stream_lstm_adaptive_...') để không ghi đè nhau


# ── Hyper-parameters — giống hệt train.py ─────────────────────────────────────
emb_dim       = 512
attention_dim = 512
decoder_dim   = 512
dropout       = 0.5
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

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
fine_tune_encoder        = True
checkpoint               = None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch
    global fine_tune_encoder, data_name

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

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
        # Dùng DecoderBaseline thay vì DecoderAdaptive
        decoder = DecoderBaseline(
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

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            print("No improvement for 20 epochs — stopping early.")
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
            print(f"Epochs since improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0

        save_checkpoint(
            checkpoint_name, epoch, epochs_since_improvement,
            encoder, decoder, encoder_optimizer, decoder_optimizer,
            recent_bleu4, is_best,
        )


# ── Train một epoch ────────────────────────────────────────────────────────────
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

        imgs, padding_mask = encoder(imgs)
        padding_mask = padding_mask.to(device)

        # DecoderBaseline.forward() signature tương thích DecoderAdaptive
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens, padding_mask
        )
        targets = caps_sorted[:, 1:]

        scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores_packed, targets_packed)

        # ── Alpha regularization ───────────────────────────────────────────────
        # alphas: [B, max_T, N]  ← N chiều (không có sentinel, nên không có N+1)
        # Công thức vẫn y chang train.py — torch tự broadcast đúng chiều cuối
        #
        # Ý nghĩa: khuyến khích mỗi spatial slot được attend ít nhất một lần
        # qua suốt quá trình decode. Với BasicAttention, "slot" chỉ là encoder
        # features (không có sentinel option), nên penalty này chặt hơn.
        max_T     = alphas.size(1)
        dl_tensor = torch.tensor(decode_lengths, dtype=torch.float, device=device)
        t_idx     = torch.arange(max_T, device=device).unsqueeze(0).float()
        time_mask = (t_idx < dl_tensor.unsqueeze(1)).float()         # [B, max_T]
        alpha_sum = (alphas * time_mask.unsqueeze(2)).sum(dim=1)     # [B, N]
        loss += alpha_c * ((1.0 - alpha_sum) ** 2).mean()

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

        top5 = accuracy(scores_packed, targets_packed, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Top-5 Acc {top5accs.val:.3f} ({top5accs.avg:.3f})'
            )


# ── Validate ───────────────────────────────────────────────────────────────────
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
    scorer       = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    skip_tokens  = {word_map['<start>'], word_map['<pad>'], word_map['<end>']}

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs    = imgs.to(device)
            caps    = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

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

            scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

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
                img_captions = [
                    [w for w in c if w not in skip_tokens]
                    for c in img_caps
                ]
                references.append(img_captions)
                references_str.append([
                    " ".join(rev_word_map.get(w, "") for w in cap)
                    for cap in img_captions
                ])

            # ── Hypotheses ─────────────────────────────────────────────────────
            _, preds = torch.max(scores_copy, dim=2)
            preds_list = [
                [w for w in preds[j][:decode_lengths[j]].tolist()
                 if w not in skip_tokens]
                for j in range(len(preds))
            ]
            hypotheses.extend(preds_list)
            for p in preds_list:
                hypotheses_str.append(
                    " ".join(rev_word_map.get(w, "") for w in p)
                )

        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0,    0,    0   ))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,  0.5,  0,    0   ))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0   ))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        meteor_scores = [
            meteor_score([r.split() for r in refs], hyp.split())
            for refs, hyp in zip(references_str, hypotheses_str)
        ]
        meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        rouge_scores = [
            max(scorer.score(r, hyp)['rougeL'].fmeasure for r in refs)
            for refs, hyp in zip(references_str, hypotheses_str)
        ]
        rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        print(f'\n* [BASELINE] VAL LOSS: {losses.avg:.3f} | TOP-5 ACC: {top5accs.avg:.3f}')
        print(
            f'* BLEU-1: {bleu1:.4f} | BLEU-2: {bleu2:.4f} | '
            f'BLEU-3: {bleu3:.4f} | BLEU-4: {bleu4:.4f}'
        )
        print(f'* METEOR: {meteor:.4f} | ROUGE-L: {rouge_l:.4f}\n')

    return bleu4


if __name__ == '__main__':
    main()
