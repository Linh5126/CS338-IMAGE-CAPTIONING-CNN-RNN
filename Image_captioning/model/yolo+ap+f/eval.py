"""
eval.py — Đánh giá mô hình Image Captioning trên tập TEST.

Kiến trúc tương thích:
  Encoder : DualStreamEncoder  (ResNet101 + YOLOv12 + SpatialFusion5D)
             → trả về (combined [B,N,2048], padding_mask [B,N])
  Decoder : DecoderAdaptive    (Sentinel Adaptive Attention LSTM)

Hai chế độ giải mã:
  1. Greedy search  — argmax mỗi bước, O(L)
  2. Beam search    — beam_size=5 (mặc định), chọn hypothesis tốt nhất

Metrics báo cáo: BLEU-1/2/3/4  ·  METEOR  ·  ROUGE-L

Cách chạy:
  python eval.py
  python eval.py --beam-size 5 --checkpoint /kaggle/working/BEST_checkpoint_...pth.tar
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.amp import autocast
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
    default='/kaggle/working/BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar',
    help='Path to BEST checkpoint (.pth.tar)',
)
parser.add_argument('--data-folder', default='/kaggle/input/datasets/llnhins/cs338l/')
parser.add_argument('--data-name',   default='flickr8k_5_cap_per_img_5_min_word_freq')
parser.add_argument('--beam-size',   type=int,   default=5,  help='1 = greedy, >1 = beam search')
parser.add_argument('--max-len',     type=int,   default=50, help='Max caption length khi sinh câu')
parser.add_argument('--workers',     type=int,   default=4)
args, _ = parser.parse_known_args()

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# Hàm decode một bước LSTM  —  Sentinel Adaptive Attention
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _step(decoder, encoder_out, h, c, prev_word_emb, padding_mask):
    """
    Một bước forward của DecoderAdaptive (không dùng teacher forcing).

    Args:
        encoder_out   : [B, N, 2048]
        h, c          : [B, decoder_dim]
        prev_word_emb : [B, embed_dim]
        padding_mask  : [B, N] bool | None

    Returns:
        log_probs : [B, vocab_size]   — log_softmax scores
        h, c      : [B, decoder_dim] — LSTM state mới
    """
    # Sentinel  s_t = sigmoid(Wx·e + Wh·h) ⊙ tanh(c)
    g_t = torch.sigmoid(
        decoder.sentinel_w_x(prev_word_emb) + decoder.sentinel_w_h(h)
    )
    s_t = g_t * torch.tanh(c)

    # Adaptive Attention (với padding mask cho YOLO slots)
    context, _ = decoder.attention(encoder_out, h, s_t, padding_mask)

    # LSTM cell
    h, c = decoder.decode_step(
        torch.cat([prev_word_emb, context], dim=1), (h, c)
    )

    # Log-softmax (không dropout trong eval)
    log_probs = F.log_softmax(decoder.fc(h), dim=1)
    return log_probs, h, c


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy Search
# ═══════════════════════════════════════════════════════════════════════════════
def greedy_decode(encoder, decoder, image, word_map, max_len):
    """
    Greedy decode: argmax mỗi bước.

    Returns:
        seq : list[int] — token ids (không kể special tokens)
    """
    start_tok = word_map['<start>']
    end_tok   = word_map['<end>']
    skip      = {start_tok, end_tok, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        encoder_out, pad_mask = encoder(image)           # [1,N,2048], [1,N]
        h, c = decoder.init_hidden_state(encoder_out)    # [1,D]

        prev = torch.tensor([start_tok], dtype=torch.long, device=device)
        seq  = []

        for _ in range(max_len):
            emb = decoder.embedding(prev)                # [1,E]
            log_probs, h, c = _step(decoder, encoder_out, h, c, emb, pad_mask)
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
    Beam search decode.

    Thuật toán:
      • Encode ảnh → [1,N,2048] expand → [k,N,2048]
      • Mỗi bước tính log-prob → cộng tích lũy → top-k
      • Beam gặp <end> → lưu vào complete_seqs
      • Trả về hypothesis điểm cao nhất

    Returns:
        seq : list[int]
    """
    k          = beam_size
    start_tok  = word_map['<start>']
    end_tok    = word_map['<end>']
    vocab_size = len(word_map)
    skip       = {start_tok, end_tok, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):

        # ── Encode ────────────────────────────────────────────────────────────
        encoder_out, pad_mask = encoder(image)           # [1,N,2048], [1,N]
        N = encoder_out.size(1)

        # Expand sang k beams
        enc  = encoder_out.expand(k, N, 2048)            # [k,N,2048]
        pm   = pad_mask.expand(k, N) if pad_mask is not None else None

        h, c = decoder.init_hidden_state(enc)             # [k,D]

        # Sequences: [[<start>]] × k,  tích lũy score = 0
        seqs   = torch.full((k, 1), start_tok, dtype=torch.long,  device=device)
        scores = torch.zeros(k,               dtype=torch.float, device=device)

        complete_seqs        = []
        complete_seqs_scores = []

        # ── Decode loop ────────────────────────────────────────────────────────
        for step in range(max_len):
            s    = seqs.size(0)                          # số beam còn sống
            prev = seqs[:, -1]                           # [s]
            emb  = decoder.embedding(prev)               # [s,E]

            log_probs, h, c = _step(
                decoder, enc[:s], h, c, emb,
                pm[:s] if pm is not None else None
            )

            # Cộng tích lũy log-prob
            all_scores = scores[:s].unsqueeze(1) + log_probs   # [s, V]

            if step == 0:
                # Bước đầu: tất cả beams như nhau → chỉ lấy top-k từ beam 0
                top_scores, top_words = all_scores[0].topk(k, dim=0)
            else:
                top_scores, top_words = all_scores.view(-1).topk(k, dim=0)

            beam_idx  = top_words // vocab_size          # beam nào
            token_idx = top_words  % vocab_size          # token nào

            # Gắn token mới vào sequence
            seqs_new = torch.cat(
                [seqs[beam_idx], token_idx.unsqueeze(1)], dim=1
            )                                            # [k, step+2]

            # Phân loại complete / incomplete
            done_mask = (token_idx == end_tok)

            for j in range(k):
                if done_mask[j]:
                    tok_list = [t for t in seqs_new[j].tolist() if t not in skip]
                    complete_seqs.append(tok_list)
                    complete_seqs_scores.append(top_scores[j].item())

            alive = ~done_mask
            inc   = alive.nonzero(as_tuple=False).squeeze(1)

            if len(inc) == 0:
                break                                    # tất cả đã kết thúc

            # Cập nhật trạng thái cho các beam còn sống
            seqs   = seqs_new[inc]
            scores = top_scores[inc]
            h      = h[beam_idx[inc]]
            c      = c[beam_idx[inc]]
            enc    = enc[beam_idx[inc]]
            if pm is not None:
                pm = pm[beam_idx[inc]]

        # ── Chọn hypothesis tốt nhất ──────────────────────────────────────────
        if complete_seqs:
            best = max(range(len(complete_seqs_scores)),
                       key=lambda i: complete_seqs_scores[i])
            seq = complete_seqs[best]
        else:
            # Fallback: không beam nào sinh ra <end> trước max_len
            seq = [t for t in seqs[0].tolist() if t not in skip]

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate loop
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(encoder, decoder, loader, word_map, beam_size, max_len):
    """
    Chạy toàn bộ tập TEST, tính BLEU / METEOR / ROUGE-L.

    Args:
        beam_size: 1 → greedy, >1 → beam search

    Returns:
        dict{'bleu1','bleu2','bleu3','bleu4','meteor','rouge_l','avg_time'}
    """
    encoder.eval()
    decoder.eval()

    rev  = {v: k for k, v in word_map.items()}
    rsc  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    skip = {word_map['<start>'], word_map['<end>'], word_map['<pad>']}

    references     = []   # [[ref1_tokens, ref2_tokens, ...], ...]  — for corpus_bleu
    hypotheses     = []   # [hyp_tokens, ...]
    references_str = []   # [[ref1_str, ...], ...]  — for METEOR / ROUGE
    hypotheses_str = []   # [hyp_str, ...]

    timer = AverageMeter()
    mode  = f'Beam k={beam_size}' if beam_size > 1 else 'Greedy'

    for image, caps, caplens, allcaps in tqdm(loader, desc=f'[{mode}]'):
        image = image.to(device)                         # [1, 3, 256, 256]

        t0 = time.time()
        if beam_size == 1:
            seq = greedy_decode(encoder, decoder, image, word_map, max_len)
        else:
            seq = beam_decode(encoder, decoder, image, word_map, beam_size, max_len)
        timer.update(time.time() - t0)

        # References — 5 captions / ảnh, lọc special tokens
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

    # ROUGE-L  — best ref per image
    rouge_scores = [
        max(rsc.score(r, hyp)['rougeL'].fmeasure for r in refs)
        for refs, hyp in zip(references_str, hypotheses_str)
    ]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        'bleu1':    bleu1,
        'bleu2':    bleu2,
        'bleu3':    bleu3,
        'bleu4':    bleu4,
        'meteor':   meteor,
        'rouge_l':  rouge_l,
        'avg_time': timer.avg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
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
    cols  = [('BLEU-1','bleu1'),('BLEU-2','bleu2'),('BLEU-3','bleu3'),
             ('BLEU-4','bleu4'),('METEOR','meteor'),('ROUGE-L','rouge_l')]
    w     = 60
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
    print(f'\n{"─"*60}')
    print(f' Checkpoint : {args.checkpoint}')
    print(f' Data       : {args.data_name}')
    print(f' Device     : {device}  |  AMP: {use_amp}')
    print(f' Beam size  : {args.beam_size}  |  Max len: {args.max_len}')
    print(f'{"─"*60}\n')

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()

    if 'bleu-4' in ckpt:
        print(f' [Checkpoint] epoch={ckpt.get("epoch","?")}  '
              f'BLEU-4={ckpt["bleu-4"]:.4f}  '
              f'METEOR={ckpt.get("meteor","?")!s:.6}  '
              f'ROUGE-L={ckpt.get("rouge-l","?")!s:.6}\n')

    # ── Word map ───────────────────────────────────────────────────────────────
    wmap_path = f'{args.data_folder}WORDMAP_{args.data_name}.json'
    with open(wmap_path, 'r') as f:
        word_map = json.load(f)
    print(f' Vocab size: {len(word_map):,}\n')

    # ── DataLoader — batch=1, shuffle=False (reproducible) ────────────────────
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    loader = torch.utils.data.DataLoader(
        CaptionDataset(
            args.data_folder,
            args.data_name,
            'TEST',
            transform=transforms.Compose([normalize]),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── [1/2] Greedy ──────────────────────────────────────────────────────────
    print('=' * 60)
    print(' [1/2] GREEDY SEARCH')
    print('=' * 60)
    g = evaluate(encoder, decoder, loader, word_map,
                 beam_size=1, max_len=args.max_len)
    _print(g, 'Greedy')

    # ── [2/2] Beam Search ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f' [2/2] BEAM SEARCH  (k = {args.beam_size})')
    print('=' * 60)
    b = evaluate(encoder, decoder, loader, word_map,
                 beam_size=args.beam_size, max_len=args.max_len)
    _print(b, f'Beam k={args.beam_size}')

    # ── Bảng so sánh ─────────────────────────────────────────────────────────
    _table(g, b, args.beam_size)


if __name__ == '__main__':
    main()
