"""
eval.py — Đánh giá mô hình Image Captioning trên tập TEST.

Hỗ trợ hai chế độ giải mã:
  1. Greedy search  (beam_size=1, nhanh, baseline)
  2. Beam search    (beam_size=k, chính xác hơn, có length penalty)

Metrics: BLEU-1/2/3/4 · METEOR · ROUGE-L · CIDEr

Tương thích với:
  - DualStreamEncoder  (encoder.forward() → (combined [B,N,2048], padding_mask [B,N]))
  - DecoderAdaptive    (Sentinel Adaptive Attention LSTM)
  - datasets.py        (CaptionDataset — tự động deduplicate unique images)
  - train_rl.py        (cùng checkpoint format, cùng word_map)

Fix so với eval cũ:
  [BUG1] Chỉ evaluate unique images (không lặp lại 5x do CaptionDataset có cpi slots)
  [BUG2] Beam search có length penalty → không thiên vị câu ngắn
  [BUG3] torch.div(..., rounding_mode='floor') thay vì // để tránh warning PyTorch ≥ 2.0
  [BUG4] Thêm CIDEr metric (model được train bằng CIDEr reward, phải report nó)
  [NEW]  --decode greedy|beam|both  để chọn chế độ
  [NEW]  --save-json  để lưu captions ra file để phân tích sau
  [NEW]  Length penalty α configurable qua --length-penalty-alpha
"""

import argparse
import json
import time
from pathlib import Path

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

# CIDEr từ pycocoevalcap (cùng thư viện dùng trong train_rl.py)
try:
    from pycocoevalcap.cider.cider import Cider
    _HAS_CIDER = True
except ImportError:
    _HAS_CIDER = False
    print("[WARN] pycocoevalcap không tìm thấy — CIDEr sẽ bị bỏ qua.")

from datasets import CaptionDataset
from utils import AverageMeter


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Image Captioning — Evaluation')
parser.add_argument('--checkpoint',
    default='/kaggle/working/BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar')
parser.add_argument('--data-folder', default='/kaggle/input/datasets/llnhins/cs338l/')
parser.add_argument('--data-name',   default='flickr8k_5_cap_per_img_5_min_word_freq')
parser.add_argument('--split',       default='TEST', choices=['TEST', 'VAL'])
parser.add_argument('--decode',      default='both', choices=['greedy', 'beam', 'both'])
parser.add_argument('--beam-size',   type=int,   default=5)
parser.add_argument('--max-len',     type=int,   default=50)
parser.add_argument('--length-penalty-alpha', type=float, default=0.7,
    help='Exponent α trong length penalty LP = ((5+len)/(5+1))^α. 0 = tắt penalty.')
parser.add_argument('--workers',     type=int,   default=4)
parser.add_argument('--save-json',   type=str,   default=None,
    help='Nếu chỉ định, lưu kết quả ra file JSON này.')
args, _ = parser.parse_known_args()

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp       = device.type == 'cuda'
cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: one decode step của DecoderAdaptive (Sentinel Adaptive Attention)
# ═══════════════════════════════════════════════════════════════════════════════
def _sentinel_step(decoder, encoder_out, h, c, prev_word_emb, padding_mask):
    """
    Một bước giải mã của DecoderAdaptive — dùng chung cho cả greedy và beam.

    Args:
        decoder       : DecoderAdaptive instance
        encoder_out   : [B, N, 2048]  — N = 49 global slots + K object slots
        h, c          : [B, decoder_dim]
        prev_word_emb : [B, embed_dim]
        padding_mask  : [B, N] bool | None  — True = padding slot (YOLO zeros)

    Returns:
        log_probs : [B, vocab_size]  — log-softmax scores (dùng để cộng tích lũy)
        h, c      : [B, decoder_dim]
    """
    # Sentinel gate — giống hệt DecoderAdaptive.forward()
    g_t = torch.sigmoid(
        decoder.sentinel_w_x(prev_word_emb) + decoder.sentinel_w_h(h)
    )
    s_t = g_t * torch.tanh(c)                                    # [B, decoder_dim]

    # Adaptive attention (có padding mask cho YOLO zero slots)
    context, _ = decoder.attention(encoder_out, h, s_t, padding_mask)

    # LSTM step — không dropout trong eval (model.eval() đã tắt)
    h, c = decoder.decode_step(
        torch.cat([prev_word_emb, context], dim=1), (h, c)
    )

    # Log-softmax để cộng log-prob (tránh underflow với multiplication)
    log_probs = F.log_softmax(decoder.fc(h), dim=1)              # [B, vocab_size]
    return log_probs, h, c


# ═══════════════════════════════════════════════════════════════════════════════
# Length penalty (Wu et al., 2016 — Google NMT)
#   LP(Y) = ((5 + |Y|) / (5 + 1)) ^ α
#   α = 0   → không penalty (raw log-prob)
#   α = 0.7 → mặc định, cân bằng giữa ngắn và dài
#   α = 1.0 → penalty mạnh nhất
# ═══════════════════════════════════════════════════════════════════════════════
def _length_penalty(length: int, alpha: float) -> float:
    if alpha == 0.0:
        return 1.0
    return ((5.0 + length) / 6.0) ** alpha


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy Search — O(L) steps, nhanh
# ═══════════════════════════════════════════════════════════════════════════════
def greedy_decode(encoder, decoder, image, word_map, max_len):
    """
    Greedy decode một ảnh (batch size = 1).

    Returns:
        seq : list[int] — token ids đã lọc (không kể <start>, <end>, <pad>)
    """
    start_token = word_map['<start>']
    end_token   = word_map['<end>']
    skip        = {start_token, end_token, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        # DualStreamEncoder trả về (combined, padding_mask)
        encoder_out, padding_mask = encoder(image)               # [1,N,2048], [1,N]
        padding_mask = padding_mask.to(device)

        h, c = decoder.init_hidden_state(encoder_out)            # [1, D]

        prev_word = torch.tensor([start_token], dtype=torch.long, device=device)
        seq = []

        for _ in range(max_len):
            emb = decoder.embedding(prev_word)                   # [1, E]
            log_probs, h, c = _sentinel_step(
                decoder, encoder_out, h, c, emb, padding_mask
            )
            token = log_probs.argmax(dim=1).item()
            if token == end_token:
                break
            if token not in skip:
                seq.append(token)
            prev_word = torch.tensor([token], dtype=torch.long, device=device)

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Beam Search — O(k × L) steps, chính xác hơn
# ═══════════════════════════════════════════════════════════════════════════════
def beam_decode(encoder, decoder, image, word_map, beam_size, max_len,
                length_penalty_alpha=0.7):
    """
    Beam search decode một ảnh (batch size = 1).

    Fix so với phiên bản cũ:
      - torch.div(..., rounding_mode='floor') thay vì //
      - Length penalty khi chọn hypothesis tốt nhất
      - Score normalization đúng trong complete_seqs

    Returns:
        seq : list[int] — token ids đã lọc (không kể <start>, <end>, <pad>)
    """
    k           = beam_size
    start_token = word_map['<start>']
    end_token   = word_map['<end>']
    vocab_size  = len(word_map)
    skip        = {start_token, end_token, word_map['<pad>']}

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        # ── Encode ────────────────────────────────────────────────────────────
        encoder_out, padding_mask = encoder(image)               # [1,N,2048], [1,N]
        padding_mask = padding_mask.to(device)
        N = encoder_out.size(1)

        # Expand sang k beams
        encoder_out_k  = encoder_out.expand(k, N, 2048)          # [k, N, 2048]
        padding_mask_k = padding_mask.expand(k, N).contiguous()  # [k, N]

        h, c = decoder.init_hidden_state(encoder_out_k)          # [k, D]

        # seqs[i] = token ids của beam i (bắt đầu bằng <start>)
        seqs       = torch.full((k, 1), start_token, dtype=torch.long, device=device)
        top_scores = torch.zeros(k, dtype=torch.float, device=device)

        # Beams hoàn thành: lưu (seq_tokens, raw_score, length) để normalize sau
        complete_seqs        = []   # list[list[int]]
        complete_seqs_scores = []   # list[float] — raw cumulative log-prob
        complete_seqs_lens   = []   # list[int]   — để tính length penalty

        # ── Decode loop ────────────────────────────────────────────────────────
        s = k   # số beams đang active
        for step in range(max_len):
            prev_word = seqs[:, -1]                              # [s]
            emb = decoder.embedding(prev_word)                   # [s, E]

            log_probs, h, c = _sentinel_step(
                decoder,
                encoder_out_k[:s],
                h[:s], c[:s],
                emb,
                padding_mask_k[:s],
            )
            # log_probs: [s, vocab_size]

            # Cộng log-prob tích lũy
            scores = top_scores[:s].unsqueeze(1) + log_probs    # [s, V]

            if step == 0:
                # Bước đầu: mọi beam giống nhau → chỉ lấy top-k từ beam 0
                top_scores_new, top_words = scores[0].topk(k, dim=0)
            else:
                top_scores_new, top_words = scores.view(-1).topk(k, dim=0)

            # Chuyển flat index → (beam_idx, token_idx)
            # FIX: dùng torch.div thay vì // để tránh FutureWarning PyTorch ≥ 2.0
            beam_idx  = torch.div(top_words, vocab_size, rounding_mode='floor')  # [k]
            token_idx = top_words % vocab_size                                    # [k]

            seqs_new = torch.cat(
                [seqs[beam_idx], token_idx.unsqueeze(1)], dim=1
            )                                                    # [k, step+2]

            # Phân loại beam: complete vs incomplete
            complete_mask   = (token_idx == end_token)
            incomplete_mask = ~complete_mask

            # Lưu các beam đã hoàn thành
            for j in range(k):
                if complete_mask[j]:
                    seq_tokens = [
                        t for t in seqs_new[j, 1:].tolist()     # bỏ <start>
                        if t not in skip
                    ]
                    complete_seqs.append(seq_tokens)
                    complete_seqs_scores.append(top_scores_new[j].item())
                    complete_seqs_lens.append(len(seq_tokens))

            # Giữ lại beams chưa xong
            inc_idx = incomplete_mask.nonzero(as_tuple=False).squeeze(1)
            if len(inc_idx) == 0:
                break                                            # tất cả đã kết thúc

            s = len(inc_idx)
            seqs          = seqs_new[inc_idx]
            top_scores    = top_scores_new[inc_idx]
            h             = h[beam_idx[inc_idx]]
            c             = c[beam_idx[inc_idx]]
            encoder_out_k = encoder_out_k[beam_idx[inc_idx]]
            padding_mask_k= padding_mask_k[beam_idx[inc_idx]]

        # ── Chọn hypothesis tốt nhất (có length penalty) ─────────────────────
        if complete_seqs:
            # Normalize bằng length penalty trước khi so sánh
            normed = [
                score / _length_penalty(length, length_penalty_alpha)
                for score, length in zip(complete_seqs_scores, complete_seqs_lens)
            ]
            best_idx = normed.index(max(normed))
            seq = complete_seqs[best_idx]
        else:
            # Fallback: không beam nào sinh <end> → beam có log-prob cao nhất
            seq = [t for t in seqs[0, 1:].tolist() if t not in skip]

    return seq


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate — chạy greedy hoặc beam trên DataLoader (unique images only)
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(encoder, decoder, loader, word_map, beam_size, max_len,
             length_penalty_alpha=0.7):
    """
    Đánh giá model trên DataLoader đã cho.

    Args:
        beam_size            : 1 = greedy, >1 = beam search
        max_len              : độ dài tối đa khi sinh câu
        length_penalty_alpha : α cho length penalty trong beam search

    Returns:
        dict với các metric và danh sách hypotheses/references thô
    """
    encoder.eval()
    decoder.eval()

    rev_word_map = {v: k for k, v in word_map.items()}
    rouge_sc     = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    references     = []    # list[list[list[int]]]  — BLEU corpus format
    hypotheses     = []    # list[list[int]]
    references_str = []    # list[list[str]]        — METEOR / ROUGE-L / CIDEr
    hypotheses_str = []    # list[str]

    skip     = {word_map['<start>'], word_map['<end>'], word_map['<pad>']}
    timing   = AverageMeter()
    desc     = f"Beam k={beam_size}" if beam_size > 1 else "Greedy"

    for image, _cap, _caplen, allcaps in tqdm(loader, desc=f"[{desc}]"):
        # image   : [1, 3, 256, 256]
        # allcaps : [1, cpi, max_cap_len]  — tất cả 5 captions của ảnh này
        image = image.to(device)

        t0 = time.time()
        if beam_size == 1:
            seq = greedy_decode(encoder, decoder, image, word_map, max_len)
        else:
            seq = beam_decode(encoder, decoder, image, word_map,
                              beam_size, max_len, length_penalty_alpha)
        timing.update(time.time() - t0)

        # ── References ────────────────────────────────────────────────────────
        # allcaps[0]: [cpi, max_cap_len] — lấy item đầu tiên vì batch_size=1
        img_caps = allcaps[0].tolist()
        img_captions_tok = [
            [w for w in c if w not in skip]
            for c in img_caps
        ]
        references.append(img_captions_tok)

        ref_strs = [
            " ".join(rev_word_map.get(w, "<unk>") for w in cap)
            for cap in img_captions_tok
        ]
        references_str.append(ref_strs)

        # ── Hypothesis ────────────────────────────────────────────────────────
        hypotheses.append(seq)
        hyp_str = " ".join(rev_word_map.get(w, "<unk>") for w in seq)
        hypotheses_str.append(hyp_str)

    assert len(references) == len(hypotheses), \
        f"Số references ({len(references)}) ≠ hypotheses ({len(hypotheses)})"

    n = len(hypotheses)

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
    meteor = sum(meteor_scores) / n if meteor_scores else 0.0

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    rouge_scores = [
        max(rouge_sc.score(r, hyp)['rougeL'].fmeasure for r in refs)
        for refs, hyp in zip(references_str, hypotheses_str)
    ]
    rouge_l = sum(rouge_scores) / n if rouge_scores else 0.0

    # ── CIDEr — dùng đúng corpus IDF (không phải per-batch như lúc train) ────
    cider = 0.0
    if _HAS_CIDER:
        # pycocoevalcap.Cider nhận dict: {img_id: [list of captions]}
        gts = {i: refs for i, refs in enumerate(references_str)}
        res = {i: [hyp]  for i, hyp  in enumerate(hypotheses_str)}
        cider_scorer = Cider()
        cider, _ = cider_scorer.compute_score(gts, res)

    return {
        'bleu1':    bleu1,
        'bleu2':    bleu2,
        'bleu3':    bleu3,
        'bleu4':    bleu4,
        'meteor':   meteor,
        'rouge_l':  rouge_l,
        'cider':    cider,
        'avg_time': timing.avg,
        'n_images': n,
        # Trả về raw data để save JSON nếu cần
        '_hypotheses_str': hypotheses_str,
        '_references_str': references_str,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'═'*62}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"  Data       : {args.data_name}")
    print(f"  Device     : {device}  |  AMP: {use_amp}")
    print(f"  Decode     : {args.decode}  |  Beam size: {args.beam_size}")
    print(f"  LP alpha   : {args.length_penalty_alpha}")
    print(f"{'═'*62}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()
    decoder = ckpt['decoder'].to(device).eval()
    print(f"  Loaded checkpoint từ epoch {ckpt.get('epoch', '?')}  "
          f"(best BLEU-4 = {ckpt.get('bleu-4', 0):.4f})\n")

    # ── Word map ───────────────────────────────────────────────────────────────
    wmap_path = Path(args.data_folder) / f"WORDMAP_{args.data_name}.json"
    with open(wmap_path, 'r') as f:
        word_map = json.load(f)
    print(f"  Vocab size : {len(word_map)}")

    # ── Dataset — chỉ lấy UNIQUE images ──────────────────────────────────────
    # BUG1 FIX: CaptionDataset có num_images * cpi items (vd 5000 cho Flickr8k test).
    # Mỗi cpi items liên tiếp dùng cùng một ảnh, chỉ khác caption làm reference.
    # Dùng Subset với bước = cpi để chỉ lấy mỗi ảnh một lần.
    # allcaps trong __getitem__ đã chứa đủ 5 captions → references vẫn đầy đủ.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    full_dataset = CaptionDataset(
        args.data_folder, args.data_name, args.split,
        transform=transforms.Compose([normalize]),
    )
    cpi            = full_dataset.cpi
    unique_indices = list(range(0, len(full_dataset), cpi))
    unique_dataset = torch.utils.data.Subset(full_dataset, unique_indices)

    loader = torch.utils.data.DataLoader(
        unique_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"  Images     : {len(unique_dataset)} unique "
          f"(full dataset = {len(full_dataset)}, cpi = {cpi})\n")

    # ── Run evaluations ───────────────────────────────────────────────────────
    results = {}

    if args.decode in ('greedy', 'both'):
        print("─" * 62)
        print("  [Greedy Search]")
        print("─" * 62)
        g = evaluate(encoder, decoder, loader, word_map,
                     beam_size=1, max_len=args.max_len,
                     length_penalty_alpha=args.length_penalty_alpha)
        _print_metrics(g, "Greedy")
        results['greedy'] = g

    if args.decode in ('beam', 'both'):
        print("─" * 62)
        print(f"  [Beam Search  k={args.beam_size}  α={args.length_penalty_alpha}]")
        print("─" * 62)
        b = evaluate(encoder, decoder, loader, word_map,
                     beam_size=args.beam_size, max_len=args.max_len,
                     length_penalty_alpha=args.length_penalty_alpha)
        _print_metrics(b, f"Beam k={args.beam_size}")
        results['beam'] = b

    # ── Bảng so sánh ─────────────────────────────────────────────────────────
    if args.decode == 'both':
        _print_comparison(results['greedy'], results['beam'], args.beam_size)

    # ── Lưu JSON nếu yêu cầu ─────────────────────────────────────────────────
    if args.save_json:
        _save_json(results, args.save_json, word_map)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print_metrics(m, title):
    print(f"\n  ── Kết quả: {title} ──")
    print(f"  BLEU-1  : {m['bleu1']:.4f}")
    print(f"  BLEU-2  : {m['bleu2']:.4f}")
    print(f"  BLEU-3  : {m['bleu3']:.4f}")
    print(f"  BLEU-4  : {m['bleu4']:.4f}")
    print(f"  METEOR  : {m['meteor']:.4f}")
    print(f"  ROUGE-L : {m['rouge_l']:.4f}")
    if _HAS_CIDER:
        print(f"  CIDEr   : {m['cider']:.4f}")
    print(f"  Thời gian/ảnh : {m['avg_time']*1000:.1f} ms")
    print(f"  Số ảnh đã eval: {m['n_images']}\n")


def _print_comparison(g, b, beam_size):
    print(f"\n{'═'*62}")
    print(f"  {'Metric':<12} {'Greedy':>10} {'Beam k='+str(beam_size):>14} {'Δ':>8}")
    print(f"  {'─'*58}")
    metrics = [
        ('bleu1',  'BLEU-1'),
        ('bleu2',  'BLEU-2'),
        ('bleu3',  'BLEU-3'),
        ('bleu4',  'BLEU-4'),
        ('meteor', 'METEOR'),
        ('rouge_l','ROUGE-L'),
    ]
    if _HAS_CIDER:
        metrics.append(('cider', 'CIDEr'))
    for key, label in metrics:
        gv = g[key]; bv = b[key]; d = bv - gv
        sign = '+' if d >= 0 else ''
        print(f"  {label:<12} {gv:>10.4f} {bv:>14.4f} {sign+f'{d:.4f}':>8}")
    print(f"  {'Time/img':<12} {g['avg_time']*1000:>9.1f}ms {b['avg_time']*1000:>13.1f}ms")
    print(f"{'═'*62}\n")


def _save_json(results, path, word_map):
    """Lưu hypotheses và references ra JSON để phân tích định tính sau."""
    output = {}
    for mode, m in results.items():
        output[mode] = {
            'metrics': {k: v for k, v in m.items() if not k.startswith('_')},
            'samples': [
                {'hypothesis': hyp, 'references': refs}
                for hyp, refs in zip(m['_hypotheses_str'], m['_references_str'])
            ],
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu kết quả ra: {path}")


if __name__ == '__main__':
    main()
