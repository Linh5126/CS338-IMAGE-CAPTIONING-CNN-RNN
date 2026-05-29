"""
eval.py — Đánh giá mô hình Image Captioning trên tập TEST.

Hỗ trợ hai chế độ giải mã:
  1. Greedy search  (nhanh, baseline)
  2. Beam search    (k=5, chính xác hơn)

Metrics: BLEU-1/2/3/4 · METEOR · ROUGE-L

Tương thích với:
  - DualStreamEncoder  (encoder trả về tuple: combined, padding_mask)
  - DecoderAdaptive    (Sentinel Adaptive Attention LSTM)
  - datasets.py        (CaptionDataset)
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Image Captioning — Evaluation')
parser.add_argument('--checkpoint',
                    default='/kaggle/working/BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar')
parser.add_argument('--data-folder',   default='/kaggle/input/datasets/llnhins/cs338l/')
parser.add_argument('--data-name',     default='flickr8k_5_cap_per_img_5_min_word_freq')
parser.add_argument('--beam-size',     type=int, default=5)
parser.add_argument('--max-len',       type=int, default=50)
parser.add_argument('--workers',       type=int, default=4)
args, _ = parser.parse_known_args()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp   = device.type == 'cuda'
cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# Decode một bước LSTM với Sentinel Adaptive Attention
# ═══════════════════════════════════════════════════════════════════════════════
def _sentinel_step(decoder, encoder_out, h, c, prev_word_emb, padding_mask):
    """
    Một bước giải mã của DecoderAdaptive.

    Args:
        decoder       : DecoderAdaptive instance
        encoder_out   : [B, N, 2048]
        h, c          : [B, decoder_dim]
        prev_word_emb : [B, embed_dim]  — embedding của token vừa sinh
        padding_mask  : [B, N] bool | None

    Returns:
        logits : [B, vocab_size]  — log_softmax scores
        h, c   : [B, decoder_dim] — trạng thái LSTM mới
    """
    # Sentinel vector
    g_t = torch.sigmoid(
        decoder.sentinel_w_x(prev_word_emb)
        + decoder.sentinel_w_h(h)
    )
    s_t = g_t * torch.tanh(c)                                # [B, decoder_dim]

    # Adaptive attention (có padding mask cho YOLO slots)
    context, _ = decoder.attention(encoder_out, h, s_t, padding_mask)

    # LSTM step
    h, c = decoder.decode_step(
        torch.cat([prev_word_emb, context], dim=1),
        (h, c),
    )

    # Logits — không dùng dropout trong eval
    logits = F.log_softmax(decoder.fc(h), dim=1)             # [B, vocab_size]
    return logits, h, c


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy Search — O(L) steps, nhanh
# ═══════════════════════════════════════════════════════════════════════════════
def greedy_decode(encoder, decoder, image, word_map, max_len):
    """
    Greedy decode một ảnh (batch size = 1).

    Returns:
        seq : list[int] — token ids (không kể <start>, <end>, <pad>)
    """
    start_token = word_map['<start>']
    end_token   = word_map['<end>']
    skip        = {start_token, end_token, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        encoder_out, padding_mask = encoder(image)           # [1, N, 2048], [1, N]
        h, c = decoder.init_hidden_state(encoder_out)        # [1, D]

        prev_word = torch.tensor([start_token], dtype=torch.long, device=device)
        seq = []

        for _ in range(max_len):
            emb    = decoder.embedding(prev_word)            # [1, E]
            logits, h, c = _sentinel_step(
                decoder, encoder_out, h, c, emb, padding_mask
            )
            chosen = logits.argmax(dim=1)                    # [1]
            token  = chosen.item()
            if token == end_token:
                break
            if token not in skip:
                seq.append(token)
            prev_word = chosen

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Beam Search — O(k × L) steps, chính xác hơn
# ═══════════════════════════════════════════════════════════════════════════════
def beam_decode(encoder, decoder, image, word_map, beam_size, max_len):
    """
    Beam search decode một ảnh (batch size = 1).

    Thuật toán:
      - Encode ảnh → encoder_out [1, N, 2048], padding_mask [1, N]
      - Expand lên k beams
      - Mỗi bước: tính log-prob, cộng tích lũy, chọn top-k
      - Khi beam gặp <end> → lưu vào complete_seqs
      - Trả về hypothesis có điểm cao nhất

    Returns:
        seq : list[int] — token ids (không kể <start>, <end>, <pad>)
    """
    k           = beam_size
    start_token = word_map['<start>']
    end_token   = word_map['<end>']
    vocab_size  = len(word_map)
    skip        = {start_token, end_token, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        # ── Encode ────────────────────────────────────────────────────────────
        encoder_out, padding_mask = encoder(image)           # [1, N, 2048], [1, N]
        N = encoder_out.size(1)

        # Expand sang k beams
        encoder_out  = encoder_out.expand(k, N, 2048)        # [k, N, 2048]
        if padding_mask is not None:
            padding_mask = padding_mask.expand(k, N)         # [k, N]

        h, c = decoder.init_hidden_state(encoder_out)        # [k, D]

        # Sequences: bắt đầu với <start>, điểm = 0
        seqs       = torch.full((k, 1), start_token, dtype=torch.long,  device=device)
        top_scores = torch.zeros(k,    dtype=torch.float, device=device)

        complete_seqs        = []
        complete_seqs_scores = []

        # ── Decode loop ────────────────────────────────────────────────────────
        for step in range(max_len):
            # Token hiện tại của mỗi beam
            prev_word = seqs[:, -1]                          # [k]
            emb = decoder.embedding(prev_word)               # [k, E]

            # Giảm beam nếu có beam kết thúc → padding_mask cũng phải theo
            cur_pm = padding_mask[:seqs.size(0)] if padding_mask is not None else None

            logits, h, c = _sentinel_step(
                decoder, encoder_out[:seqs.size(0)], h, c, emb, cur_pm
            )
            # logits: [s, vocab_size]  (log_softmax đã áp dụng)

            # Cộng tích lũy log-prob
            scores = top_scores[:seqs.size(0)].unsqueeze(1) + logits   # [s, V]

            if step == 0:
                # Bước đầu: tất cả beams giống nhau, chỉ lấy top-k từ beam 0
                top_scores_new, top_words = scores[0].topk(k, dim=0)
            else:
                top_scores_new, top_words = scores.view(-1).topk(k, dim=0)

            # Chuyển từ chỉ số unrolled → beam_idx và token_idx
            beam_idx  = top_words // vocab_size              # [k]
            token_idx = top_words  % vocab_size              # [k]

            # Cập nhật sequences
            seqs_new = torch.cat(
                [seqs[beam_idx], token_idx.unsqueeze(1)], dim=1
            )                                                # [k, step+2]

            # Phân loại: complete (gặp <end>) hay incomplete
            incomplete_mask = (token_idx != end_token)
            complete_mask   = ~incomplete_mask

            # Lưu các beam đã hoàn thành
            for j in range(k):
                if complete_mask[j]:
                    seq_tokens = [
                        t for t in seqs_new[j].tolist()
                        if t not in skip
                    ]
                    complete_seqs.append(seq_tokens)
                    complete_seqs_scores.append(top_scores_new[j].item())

            # Giữ lại các beam chưa xong
            inc_idx = incomplete_mask.nonzero(as_tuple=False).squeeze(1)
            if len(inc_idx) == 0:
                break                                        # tất cả đã kết thúc

            seqs       = seqs_new[inc_idx]
            top_scores = top_scores_new[inc_idx]
            h          = h[beam_idx[inc_idx]]
            c          = c[beam_idx[inc_idx]]
            encoder_out  = encoder_out[beam_idx[inc_idx]]
            if padding_mask is not None:
                padding_mask = padding_mask[beam_idx[inc_idx]]

        # ── Chọn hypothesis tốt nhất ──────────────────────────────────────────
        if complete_seqs:
            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[best_idx]
        else:
            # Fallback: không beam nào sinh ra <end> → dùng beam có điểm cao nhất
            seq = [
                t for t in seqs[0].tolist()
                if t not in skip
            ]

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate — chạy cả greedy lẫn beam trên tập TEST
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(encoder, decoder, loader, word_map, beam_size, max_len):
    """
    Đánh giá model trên DataLoader đã cho.

    Args:
        beam_size : int — 1 = greedy, >1 = beam search
        max_len   : int — độ dài tối đa khi sinh câu

    Returns:
        dict với các metric: bleu1/2/3/4, meteor, rouge_l
    """
    encoder.eval()
    decoder.eval()

    rev_word_map = {v: k for k, v in word_map.items()}
    rouge_sc     = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    references     = []      # list[list[list[int]]]  — BLEU corpus format
    hypotheses     = []      # list[list[int]]
    references_str = []      # list[list[str]]        — METEOR / ROUGE-L
    hypotheses_str = []      # list[str]

    skip = {word_map['<start>'], word_map['<end>'], word_map['<pad>']}

    timing = AverageMeter()

    desc = f"Beam={beam_size}" if beam_size > 1 else "Greedy"
    for image, caps, caplens, allcaps in tqdm(loader, desc=f"Evaluating [{desc}]"):
        image = image.to(device)                             # [1, 3, 256, 256]

        t0 = time.time()
        if beam_size == 1:
            seq = greedy_decode(encoder, decoder, image, word_map, max_len)
        else:
            seq = beam_decode(encoder, decoder, image, word_map, beam_size, max_len)
        timing.update(time.time() - t0)

        # ── References ────────────────────────────────────────────────────────
        img_caps = allcaps[0].tolist()                       # list of 5 encoded captions
        img_captions = [
            [w for w in c if w not in skip]
            for c in img_caps
        ]
        references.append(img_captions)
        references_str.append([
            " ".join(rev_word_map.get(w, "") for w in cap)
            for cap in img_captions
        ])

        # ── Hypothesis ────────────────────────────────────────────────────────
        hypotheses.append(seq)
        hypotheses_str.append(" ".join(rev_word_map.get(w, "") for w in seq))

    assert len(references) == len(hypotheses)

    # ── BLEU ──────────────────────────────────────────────────────────────────
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,    0,    0,    0   ))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,  0.5,  0,    0   ))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0   ))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # ── METEOR ────────────────────────────────────────────────────────────────
    meteor_scores = [
        meteor_score([r.split() for r in refs], hyp.split())
        for refs, hyp in zip(references_str, hypotheses_str)
    ]
    meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    rouge_scores = [
        max(rouge_sc.score(r, hyp)['rougeL'].fmeasure for r in refs)
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
        'avg_time': timing.avg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'─'*60}")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Data       : {args.data_name}")
    print(f" Device     : {device}")
    print(f" AMP        : {use_amp}")
    print(f" Beam size  : {args.beam_size}")
    print(f"{'─'*60}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()

    # ── Word map ───────────────────────────────────────────────────────────────
    wmap_path = f"{args.data_folder}WORDMAP_{args.data_name}.json"
    with open(wmap_path, 'r') as f:
        word_map = json.load(f)

    # ── DataLoader (batch=1, shuffle=False để kết quả reproducible) ───────────
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

    # ── Greedy ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(" [1/2] GREEDY SEARCH")
    print("=" * 60)
    g_metrics = evaluate(encoder, decoder, loader, word_map,
                         beam_size=1, max_len=args.max_len)
    _print_metrics(g_metrics, "Greedy")

    # ── Beam Search ───────────────────────────────────────────────────────────
    print("=" * 60)
    print(f" [2/2] BEAM SEARCH  (k = {args.beam_size})")
    print("=" * 60)
    b_metrics = evaluate(encoder, decoder, loader, word_map,
                         beam_size=args.beam_size, max_len=args.max_len)
    _print_metrics(b_metrics, f"Beam k={args.beam_size}")

    # ── So sánh ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f" {'Metric':<12} {'Greedy':>10} {'Beam k='+str(args.beam_size):>12} {'Δ':>8}")
    print("-" * 60)
    for key, label in [
        ('bleu1',  'BLEU-1'),
        ('bleu2',  'BLEU-2'),
        ('bleu3',  'BLEU-3'),
        ('bleu4',  'BLEU-4'),
        ('meteor', 'METEOR'),
        ('rouge_l','ROUGE-L'),
    ]:
        g_val = g_metrics[key]
        b_val = b_metrics[key]
        delta = b_val - g_val
        sign  = '+' if delta >= 0 else ''
        print(f" {label:<12} {g_val:>10.4f} {b_val:>12.4f} {sign+f'{delta:.4f}':>8}")
    print(f" {'Time/img':<12} {g_metrics['avg_time']*1000:>9.1f}ms {b_metrics['avg_time']*1000:>11.1f}ms")
    print("=" * 60)


def _print_metrics(m, title):
    print(f"\n Results — {title}")
    print(f"  BLEU-1 : {m['bleu1']:.4f}")
    print(f"  BLEU-2 : {m['bleu2']:.4f}")
    print(f"  BLEU-3 : {m['bleu3']:.4f}")
    print(f"  BLEU-4 : {m['bleu4']:.4f}")
    print(f"  METEOR : {m['meteor']:.4f}")
    print(f"  ROUGE-L: {m['rouge_l']:.4f}")
    print(f"  Avg time/img: {m['avg_time']*1000:.1f} ms\n")


if __name__ == '__main__':
    main()
