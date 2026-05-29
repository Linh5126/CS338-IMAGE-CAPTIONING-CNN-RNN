"""
eval.py — Đánh giá mô hình Image Captioning trên tập TEST.

Kiến trúc tương thích:
  Encoder : Encoder  (ResNet101 → [B, enc², enc², 2048])
  Decoder : DecoderWithAttention  (Soft Attention + gating + LSTM)

Hai chế độ: Greedy search  ·  Beam search (k=5 mặc định)
Metrics   : BLEU-1/2/3/4  ·  METEOR  ·  ROUGE-L

Cách chạy:
  python eval.py
  python eval.py --beam-size 5 --checkpoint /content/drive/MyDrive/CS338/BEST_...pth.tar
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

import nltk
nltk.download('wordnet', quiet=True)
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from datasets import CaptionDataset
from utils import AverageMeter


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Image Captioning — Evaluation')
parser.add_argument(
    '--checkpoint',
    default='/content/drive/MyDrive/CS338/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar',
)
parser.add_argument('--data-folder', default='/content/drive/MyDrive/CS338')
parser.add_argument('--data-name',   default='flickr8k_5_cap_per_img_5_min_word_freq')
parser.add_argument('--beam-size',   type=int, default=5)
parser.add_argument('--max-len',     type=int, default=50)
parser.add_argument('--workers',     type=int, default=1)
args, _ = parser.parse_known_args()

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# Một bước decode  —  Soft Attention + sigmoid gate (f_beta)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _step(decoder, encoder_out, h, c, prev_word_emb):
    """
    Args:
        encoder_out   : [B, num_pixels, 2048]
        h, c          : [B, decoder_dim]
        prev_word_emb : [B, embed_dim]
    Returns:
        log_probs : [B, vocab_size]
        h, c      : [B, decoder_dim]
    """
    awe, _  = decoder.attention(encoder_out, h)        # soft attention
    gate    = decoder.sigmoid(decoder.f_beta(h))        # gating scalar
    awe     = gate * awe
    h, c    = decoder.decode_step(
        torch.cat([prev_word_emb, awe], dim=1), (h, c)
    )
    # Không dùng dropout trong eval
    log_probs = F.log_softmax(decoder.fc(h), dim=1)
    return log_probs, h, c


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy Search
# ═══════════════════════════════════════════════════════════════════════════════
def greedy_decode(encoder, decoder, image, word_map, max_len):
    """Greedy: chọn argmax mỗi bước. Returns list[int] (không kể special tokens)."""
    start_tok = word_map['<start>']
    end_tok   = word_map['<end>']
    skip      = {start_tok, end_tok, word_map['<pad>']}

    with torch.no_grad():
        enc_out  = encoder(image)                          # [1, 14, 14, 2048]
        enc_dim  = enc_out.size(-1)
        enc_out  = enc_out.view(1, -1, enc_dim)            # [1, 196, 2048]
        h, c     = decoder.init_hidden_state(enc_out)      # [1, D]

        prev = torch.tensor([start_tok], dtype=torch.long, device=device)
        seq  = []

        for _ in range(max_len):
            emb = decoder.embedding(prev)                  # [1, E]
            log_probs, h, c = _step(decoder, enc_out, h, c, emb)
            token = log_probs.argmax(dim=1).item()
            if token == end_tok:
                break
            if token not in skip:
                seq.append(token)
            prev = torch.tensor([token], dtype=torch.long, device=device)

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Beam Search
# ═══════════════════════════════════════════════════════════════════════════════
def beam_decode(encoder, decoder, image, word_map, beam_size, max_len):
    """
    Beam search. Returns list[int] hypothesis tốt nhất.

    Fixes so với eval.py cũ:
      • `top_words // vocab_size`  — integer division (cũ dùng `/` → float, sai index)
      • h, c, enc_out re-index đúng theo beam_idx[inc] sau mỗi bước
      • Fallback khi không có complete sequence (không crash)
    """
    k          = beam_size
    start_tok  = word_map['<start>']
    end_tok    = word_map['<end>']
    vocab_size = len(word_map)
    skip       = {start_tok, end_tok, word_map['<pad>']}

    with torch.no_grad():
        # Encode
        enc_out = encoder(image)                           # [1, 14, 14, 2048]
        enc_dim = enc_out.size(-1)
        enc_out = enc_out.view(1, -1, enc_dim)             # [1, 196, 2048]
        num_pix = enc_out.size(1)

        # Expand → k beams
        enc_out = enc_out.expand(k, num_pix, enc_dim)      # [k, 196, 2048]
        h, c    = decoder.init_hidden_state(enc_out)        # [k, D]

        seqs   = torch.full((k, 1), start_tok, dtype=torch.long,  device=device)
        scores = torch.zeros(k,               dtype=torch.float, device=device)

        complete_seqs        = []
        complete_seqs_scores = []

        for step in range(max_len):
            s    = seqs.size(0)
            prev = seqs[:, -1]                             # [s]
            emb  = decoder.embedding(prev)                 # [s, E]

            log_probs, h, c = _step(decoder, enc_out[:s], h, c, emb)

            # Cộng tích lũy log-prob
            all_scores = scores[:s].unsqueeze(1) + log_probs    # [s, V]

            if step == 0:
                top_scores, top_words = all_scores[0].topk(k, dim=0)
            else:
                top_scores, top_words = all_scores.view(-1).topk(k, dim=0)

            # FIX: integer division
            beam_idx  = top_words // vocab_size
            token_idx = top_words  % vocab_size

            seqs_new = torch.cat(
                [seqs[beam_idx], token_idx.unsqueeze(1)], dim=1
            )

            # Tách complete / incomplete
            done_mask = (token_idx == end_tok)
            for j in range(k):
                if done_mask[j]:
                    tok_list = [t for t in seqs_new[j].tolist() if t not in skip]
                    complete_seqs.append(tok_list)
                    complete_seqs_scores.append(top_scores[j].item())

            alive = ~done_mask
            inc   = alive.nonzero(as_tuple=False).squeeze(1)
            if len(inc) == 0:
                break

            # Cập nhật trạng thái beams còn sống
            seqs    = seqs_new[inc]
            scores  = top_scores[inc]
            h       = h[beam_idx[inc]]
            c       = c[beam_idx[inc]]
            enc_out = enc_out[beam_idx[inc]]

        # Chọn hypothesis tốt nhất
        if complete_seqs:
            best = max(range(len(complete_seqs_scores)),
                       key=lambda i: complete_seqs_scores[i])
            seq = complete_seqs[best]
        else:
            # Fallback: không beam nào sinh <end> → lấy beam đầu
            seq = [t for t in seqs[0].tolist() if t not in skip]

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate loop
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(encoder, decoder, loader, word_map, beam_size, max_len):
    encoder.eval()
    decoder.eval()

    rev  = {v: k for k, v in word_map.items()}
    rsc  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    skip = {word_map['<start>'], word_map['<end>'], word_map['<pad>']}

    references     = []
    hypotheses     = []
    references_str = []
    hypotheses_str = []
    timer          = AverageMeter()

    mode = f'Beam k={beam_size}' if beam_size > 1 else 'Greedy'

    for image, caps, caplens, allcaps in tqdm(loader, desc=f'[{mode}]'):
        image = image.to(device)

        t0 = time.time()
        seq = (greedy_decode(encoder, decoder, image, word_map, max_len)
               if beam_size == 1
               else beam_decode(encoder, decoder, image, word_map, beam_size, max_len))
        timer.update(time.time() - t0)

        # References — 5 captions mỗi ảnh
        img_caps = allcaps[0].tolist()
        ref_tok  = [[w for w in c if w not in skip] for c in img_caps]
        ref_str  = [' '.join(rev.get(w, '') for w in cap) for cap in ref_tok]
        references.append(ref_tok)
        references_str.append(ref_str)

        # Hypothesis
        hyp_str = ' '.join(rev.get(w, '') for w in seq)
        hypotheses.append(seq)
        hypotheses_str.append(hyp_str)

    assert len(references) == len(hypotheses)

    # BLEU
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,    0,    0,    0   ))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,  0.5,  0,    0   ))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0   ))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # METEOR
    met_scores = [
        meteor_score([r.split() for r in refs], hyp.split())
        for refs, hyp in zip(references_str, hypotheses_str)
    ]
    meteor = sum(met_scores) / len(met_scores) if met_scores else 0.0

    # ROUGE-L (best ref mỗi ảnh)
    rouge_scores = [
        max(rsc.score(r, hyp)['rougeL'].fmeasure for r in refs)
        for refs, hyp in zip(references_str, hypotheses_str)
    ]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4,
        'meteor': meteor, 'rouge_l': rouge_l, 'avg_time': timer.avg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _print(m, title):
    print(f'\n  ── {title} ──')
    print(f'  BLEU-1  : {m["bleu1"]:.4f}')
    print(f'  BLEU-2  : {m["bleu2"]:.4f}')
    print(f'  BLEU-3  : {m["bleu3"]:.4f}')
    print(f'  BLEU-4  : {m["bleu4"]:.4f}')
    print(f'  METEOR  : {m["meteor"]:.4f}')
    print(f'  ROUGE-L : {m["rouge_l"]:.4f}')
    print(f'  Time/img: {m["avg_time"]*1000:.1f} ms')


def _table(gm, bm, k):
    cols = [('BLEU-1','bleu1'),('BLEU-2','bleu2'),('BLEU-3','bleu3'),
            ('BLEU-4','bleu4'),('METEOR','meteor'),('ROUGE-L','rouge_l')]
    w = 58
    print('\n' + '='*w)
    print(f'  {"Metric":<12} {"Greedy":>10} {f"Beam k={k}":>12} {"Δ":>9}')
    print('-'*w)
    for label, key in cols:
        g, b  = gm[key], bm[key]
        delta = b - g
        sign  = '+' if delta >= 0 else ''
        print(f'  {label:<12} {g:>10.4f} {b:>12.4f} {sign+f"{delta:.4f}":>9}')
    print(f'  {"Time/img":<12} {gm["avg_time"]*1000:>9.1f}ms '
          f'{bm["avg_time"]*1000:>11.1f}ms')
    print('='*w)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f'\n{"─"*58}')
    print(f' Checkpoint : {args.checkpoint}')
    print(f' Data       : {args.data_name}')
    print(f' Device     : {device}')
    print(f' Beam size  : {args.beam_size}  |  Max len: {args.max_len}')
    print(f'{"─"*58}\n')

    # Load checkpoint
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()

    if 'bleu-4' in ckpt:
        print(f' [Checkpoint] epoch={ckpt.get("epoch","?")}  '
              f'BLEU-4={ckpt["bleu-4"]:.4f}\n')

    # Word map
    wmap_path = os.path.join(args.data_folder, f'WORDMAP_{args.data_name}.json')
    with open(wmap_path, 'r') as f:
        word_map = json.load(f)
    print(f' Vocab size: {len(word_map):,}\n')

    # DataLoader — batch=1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
        CaptionDataset(
            args.data_folder, args.data_name, 'TEST',
            transform=transforms.Compose([normalize]),
        ),
        batch_size=1, shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )

    # [1/2] Greedy
    print('='*58)
    print(' [1/2] GREEDY SEARCH')
    print('='*58)
    g = evaluate(encoder, decoder, loader, word_map,
                 beam_size=1, max_len=args.max_len)
    _print(g, 'Greedy')

    # [2/2] Beam Search
    print('\n' + '='*58)
    print(f' [2/2] BEAM SEARCH  (k = {args.beam_size})')
    print('='*58)
    b = evaluate(encoder, decoder, loader, word_map,
                 beam_size=args.beam_size, max_len=args.max_len)
    _print(b, f'Beam k={args.beam_size}')

    # Bảng so sánh
    _table(g, b, args.beam_size)


if __name__ == '__main__':
    main()
