"""
models_baseline.py — Phiên bản BASELINE để so sánh với models.py chính.

Mục đích: kiểm tra xem cơ chế Sentinel Adaptive Attention (Lu et al., 2017)
có thực sự giúp ích so với Bahdanau cross-attention thuần không, khi giữ
nguyên mọi thứ khác (encoder, dataset, hyperparameter, training loop).

Thay đổi duy nhất so với models.py:
  ✗ AdaptiveAttention (sentinel gate, softmax over N+1 slots)
  ✓ BasicAttention    (Bahdanau additive, softmax over N slots)

  ✗ DecoderAdaptive   (có sentinel_w_x, sentinel_w_h, s_t)
  ✓ DecoderBaseline   (không có sentinel — giống sgrvinod/a-PyTorch-Tutorial)

DualStreamEncoder giữ NGUYÊN VẸN — cùng ResNet101 + YOLO12s + ROI Align
để biến encoder không phải biến số trong thí nghiệm so sánh.

Signature của decoder.forward() HOÀN TOÀN TƯƠNG THÍCH với DecoderAdaptive:
  forward(encoder_out, encoded_captions, caption_lengths, padding_mask=None)
  → (predictions, encoded_captions, decode_lengths, alphas, sort_ind)

Điểm khác biệt về tensor shape:
  AdaptiveAttention : alphas [B, T, N+1]  (N slots + 1 sentinel)
  BasicAttention    : alphas [B, T, N]    (N slots, không có sentinel)
→ train_baseline.py xử lý alpha_c regularization trên N chiều thay vì N+1.
"""

import torch
from torch import nn
import torchvision
from ultralytics import YOLO
import torchvision.ops as ops

TOP_K_OBJECTS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# DualStreamEncoder — KHÔNG thay đổi so với models.py
# ═══════════════════════════════════════════════════════════════════════════════
class DualStreamEncoder(nn.Module):
    """
    Copy nguyên xi từ models.py — không chỉnh sửa bất kỳ dòng nào.
    Encoder phải giống nhau giữa hai phiên bản để kết quả so sánh công bằng.
    """

    def __init__(self, encoded_image_size=14, yolo_version='yolo12s.pt'):
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.yolo_version   = yolo_version

        resnet = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT
        )
        modules      = list(resnet.children())[:-2]
        self.resnet  = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        yolo_instance    = YOLO(self.yolo_version)
        self.yolo_core   = yolo_instance.model
        for param in self.yolo_core.parameters():
            param.requires_grad = False
        self._yolo_wrapper = [yolo_instance]

        yolo_feat_channels = self._probe_yolo_channels(yolo_instance)
        print(f"[DualStreamEncoder] YOLO feature channels detected: {yolo_feat_channels}")

        self.yolo_proj = nn.Linear(yolo_feat_channels, 2048)
        nn.init.xavier_uniform_(self.yolo_proj.weight)
        nn.init.zeros_(self.yolo_proj.bias)

        self.global_type_emb = nn.Parameter(torch.zeros(1, 1, 2048))
        self.local_type_emb  = nn.Parameter(torch.zeros(1, 1, 2048))
        nn.init.normal_(self.global_type_emb, std=0.01)
        nn.init.normal_(self.local_type_emb,  std=0.01)

    @staticmethod
    def _probe_yolo_channels(yolo_instance):
        features = []
        handle   = yolo_instance.model.model[-2].register_forward_hook(
            lambda m, inp, o: features.append(o)
        )
        try:
            with torch.no_grad():
                yolo_instance.model(torch.zeros(1, 3, 256, 256))
        finally:
            handle.remove()
        f = features[0]
        if isinstance(f, (list, tuple)):
            f = f[1] if len(f) > 1 else f[0]
        return f.shape[1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_yolo_wrapper', None)
        if '_modules' in state:
            state['_modules'] = state['_modules'].copy()
            state['_modules'].pop('yolo_core', None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        from ultralytics import YOLO as _YOLO
        _version      = getattr(self, 'yolo_version', 'yolo12s.pt')
        yolo_instance = _YOLO(_version)
        _device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_instance.to(_device)
        self.yolo_core = yolo_instance.model
        for param in self.yolo_core.parameters():
            param.requires_grad = False
        self._yolo_wrapper = [yolo_instance]

    def _parse_yolo12_boxes_batched(self, raw_output, batch_size, img_shape,
                                    device, conf_thresh=0.25):
        H, W    = img_shape
        results = [None] * batch_size
        try:
            preds = None
            if isinstance(raw_output, dict):
                for key in ('one2one', 'one2many', 'pred'):
                    if key in raw_output:
                        preds = raw_output[key]
                        break
                if preds is None:
                    preds = next(iter(raw_output.values()), None)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
            elif isinstance(raw_output, (list, tuple)):
                preds = raw_output[0]
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
            elif isinstance(raw_output, torch.Tensor):
                preds = raw_output

            if preds is None or not isinstance(preds, torch.Tensor):
                return results
            if preds.dim() == 2:
                preds = preds.unsqueeze(0)
            if preds.dim() != 3:
                return results
            if preds.shape[1] <= 6 and preds.shape[2] > 6:
                preds = preds.permute(0, 2, 1).contiguous()

            for i in range(min(preds.shape[0], batch_size)):
                p = preds[i]
                if p.shape[-1] < 5:
                    continue
                mask = p[:, 4] > conf_thresh
                p    = p[mask]
                if len(p) == 0:
                    continue
                p  = p[p[:, 4].argsort(descending=True)]
                cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                boxes = torch.stack([
                    (cx - w / 2).clamp(0, W),
                    (cy - h / 2).clamp(0, H),
                    (cx + w / 2).clamp(0, W),
                    (cy + h / 2).clamp(0, H),
                ], dim=1).to(device)
                results[i] = boxes
        except Exception as e:
            print(f"[WARN] _parse_yolo12_boxes_batched failed: {e}")
        return results

    def forward(self, images):
        device     = images.device
        batch_size = images.size(0)

        global_features = self.resnet(images)
        global_features = self.adaptive_pool(global_features)
        global_features = global_features.permute(0, 2, 3, 1).contiguous()
        global_features = global_features.view(batch_size, -1, 2048)
        global_features = global_features + self.global_type_emb

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        yolo_images = torch.clamp(images * std + mean, 0.0, 1.0)

        hook_feats = []
        handle = self.yolo_core.model[-2].register_forward_hook(
            lambda m, inp, o: hook_feats.append(o)
        )
        with torch.no_grad():
            raw_output = self.yolo_core(yolo_images)
        handle.remove()

        f_map = hook_feats[0]
        if isinstance(f_map, (list, tuple)):
            f_map = f_map[1] if len(f_map) > 1 else f_map[0]
        f_map = f_map.detach().float()

        H, W          = yolo_images.shape[-2:]
        spatial_scale = float(f_map.shape[-1]) / W

        boxes_per_img = self._parse_yolo12_boxes_batched(
            raw_output, batch_size, (H, W), device
        )

        roi_list  = []
        n_per_img = []
        for boxes in boxes_per_img:
            if boxes is not None and len(boxes) > 0:
                b = boxes[:TOP_K_OBJECTS]
            else:
                b = torch.zeros(0, 4, device=device)
            roi_list.append(b)
            n_per_img.append(len(b))

        obj_vecs   = torch.zeros(batch_size, TOP_K_OBJECTS, 2048, device=device)
        is_padding = torch.ones(batch_size,  TOP_K_OBJECTS, dtype=torch.bool, device=device)

        if any(n > 0 for n in n_per_img):
            roi_out = ops.roi_align(
                f_map, roi_list,
                output_size=(1, 1),
                spatial_scale=spatial_scale,
            )
            idx = 0
            for i, n in enumerate(n_per_img):
                if n > 0:
                    vecs = roi_out[idx : idx + n].view(n, -1)
                    vecs = self.yolo_proj(vecs)
                    obj_vecs[i, :n]   = vecs
                    is_padding[i, :n] = False
                idx += n

        real = ~is_padding
        if real.any():
            obj_vecs[real] = obj_vecs[real] + self.local_type_emb.view(2048)

        combined   = torch.cat([global_features, obj_vecs], dim=1)
        global_pad = torch.zeros(
            batch_size, global_features.size(1), dtype=torch.bool, device=device
        )
        full_pad_mask = torch.cat([global_pad, is_padding], dim=1)

        return combined, full_pad_mask

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.yolo_proj.parameters():
            p.requires_grad = True
        if fine_tune:
            for child in list(self.resnet.children())[5:]:
                for p in child.parameters():
                    p.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# BasicAttention — Bahdanau additive attention, KHÔNG có sentinel
# ═══════════════════════════════════════════════════════════════════════════════
class BasicAttention(nn.Module):
    """
    Additive (Bahdanau) attention — phiên bản gốc của Show, Attend and Tell.
    Giống hệt sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, chỉ thêm
    padding_mask để loại bỏ YOLO zero-slots khỏi softmax.

    So sánh với AdaptiveAttention trong models.py:
      AdaptiveAttention : softmax trên [N encoder slots + 1 sentinel] → alpha [B, N+1]
                          decoder tự học khi nào nên nhìn ảnh, khi nào nhìn sentinel
      BasicAttention    : softmax trên [N encoder slots] → alpha [B, N]
                          luôn luôn attend vào ảnh, không có option "bỏ qua ảnh"

    Args:
        encoder_dim   : chiều của encoder output (2048 với ResNet101)
        decoder_dim   : chiều của hidden state LSTM (512)
        attention_dim : chiều của attention space (512)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        # Chiếu encoder output lên attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Chiếu decoder hidden state lên attention space
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # Tính scalar score từ attention space
        self.full_att    = nn.Linear(attention_dim, 1)
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, padding_mask=None):
        """
        Args:
            encoder_out   : [B, N, encoder_dim]  — N = grid slots + object slots
            decoder_hidden: [B, decoder_dim]
            padding_mask  : [B, N] bool — True = YOLO padding slot (masked to -inf)

        Returns:
            context : [B, encoder_dim]  — attention-weighted encoder output
            alpha   : [B, N]            — attention weights (sum ≈ 1 trên N slots)

        Note: alpha.shape[2] = N, KHÔNG PHẢI N+1 (không có sentinel slot).
        → train_baseline.py alpha_c regularization dùng N chiều này.
        """
        # [B, N, att_dim] + [B, 1, att_dim] → broadcast → [B, N, att_dim]
        att1 = self.encoder_att(encoder_out)                        # [B, N, att_dim]
        att2 = self.decoder_att(decoder_hidden)                     # [B, att_dim]
        att  = self.full_att(
            self.relu(att1 + att2.unsqueeze(1))
        ).squeeze(2)                                                 # [B, N]

        # Mask các YOLO padding slot (vector zero) trước khi softmax
        # → tránh attention bị phân tán vào những slot không có thật
        if padding_mask is not None:
            att = att.masked_fill(padding_mask, float('-inf'))

        alpha   = self.softmax(att)                                  # [B, N]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)     # [B, encoder_dim]

        return context, alpha


# ═══════════════════════════════════════════════════════════════════════════════
# DecoderBaseline — LSTM decoder với BasicAttention, không có Sentinel
# ═══════════════════════════════════════════════════════════════════════════════
class DecoderBaseline(nn.Module):
    """
    LSTM decoder với Bahdanau cross-attention thuần — phiên bản baseline
    để so sánh với DecoderAdaptive (sentinel) trong models.py.

    Kiến trúc giống sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning:
      - Mỗi bước t: context_t = Attention(encoder_out, h_{t-1})
      - LSTM input: [embedding_t || context_t]
      - Không có sentinel gate, không có s_t

    Điểm khác biệt chính so với DecoderAdaptive:
      DecoderAdaptive  | DecoderBaseline
      ─────────────────┼──────────────────────────────────────────────────
      sentinel_w_x ✓   | ✗ — bỏ hẳn
      sentinel_w_h ✓   | ✗ — bỏ hẳn
      g_t = sigmoid(.) | ✗ — không có gate
      s_t = g_t*tanh(c)| ✗ — không có sentinel vector
      alpha [B, N+1]   | alpha [B, N]
      Attention nhận   | Attention nhận (encoder_out, h, padding_mask)
        (enc, h, s, pm)|   → không truyền sentinel

    Forward signature TƯƠNG THÍCH 100% với DecoderAdaptive:
      forward(encoder_out, encoded_captions, caption_lengths, padding_mask=None)
      → (predictions, encoded_captions, decode_lengths, alphas, sort_ind)
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.decoder_dim   = decoder_dim
        self.vocab_size    = vocab_size
        self.dropout_p     = dropout

        # BasicAttention thay vì AdaptiveAttention
        self.attention = BasicAttention(encoder_dim, decoder_dim, attention_dim)

        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.dropout     = nn.Dropout(p=dropout)

        # LSTMCell input = embed_dim + encoder_dim (giống DecoderAdaptive)
        # Không thay đổi để đảm bảo số tham số decoder tương đương
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # Init h, c từ mean encoder output (giống DecoderAdaptive)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Output projection
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # KHÔNG CÓ sentinel_w_x và sentinel_w_h — đây là điểm khác biệt duy nhất

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo giống DecoderAdaptive để công bằng khi so sánh."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Khởi tạo h, c từ mean pooling của encoder output.
        Giống hệt DecoderAdaptive.init_hidden_state().
        """
        mean_encoder_out = encoder_out.mean(dim=1)                  # [B, encoder_dim]
        h = self.init_h(mean_encoder_out)                           # [B, decoder_dim]
        c = self.init_c(mean_encoder_out)                           # [B, decoder_dim]
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths,
                padding_mask=None):
        """
        Teacher-forcing forward pass (dùng cho XE training và validation).

        Args:
            encoder_out      : [B, N, encoder_dim]   — từ DualStreamEncoder
            encoded_captions : [B, max_len]           — token ids (có <start>, <end>)
            caption_lengths  : [B, 1]                 — độ dài thực của mỗi caption
            padding_mask     : [B, N] bool | None     — True = YOLO zero slot

        Returns:
            predictions  : [B, max_t, vocab_size]
            encoded_captions: [B, max_len]            — đã sort theo length
            decode_lengths: list[int]                 — độ dài decode thực tế
            alphas       : [B, max_t, N]              — attention weights
                           ⚠ N chiều, KHÔNG PHẢI N+1 (không có sentinel)
            sort_ind     : LongTensor [B]             — sort indices
        """
        batch_size = encoder_out.size(0)
        num_slots  = encoder_out.size(1)                            # N = grid + objects

        # Sort theo caption length giảm dần (yêu cầu của pack_padded_sequence)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        if padding_mask is not None:
            padding_mask = padding_mask[sort_ind]

        embeddings     = self.embedding(encoded_captions)           # [B, max_len, embed_dim]
        h, c           = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()             # bỏ <end> khỏi target
        max_t          = max(decode_lengths)

        predictions = torch.zeros(
            batch_size, max_t, self.vocab_size, device=encoder_out.device
        )
        # alphas: [B, max_t, N] — N chiều, không phải N+1
        # Đây là điểm khác biệt về tensor shape so với DecoderAdaptive
        alphas = torch.zeros(
            batch_size, max_t, num_slots, device=encoder_out.device
        )

        # Precompute batch_size_t cho mỗi timestep (vectorized)
        dl_t          = torch.tensor(decode_lengths, device=encoder_out.device)
        t_range       = torch.arange(max_t,          device=encoder_out.device)
        batch_sizes_t = (dl_t.unsqueeze(1) > t_range.unsqueeze(0)).sum(0).tolist()

        for t in range(max_t):
            batch_size_t = batch_sizes_t[t]
            pm = padding_mask[:batch_size_t] if padding_mask is not None else None

            # BasicAttention: (encoder_out, h, padding_mask)
            # KHÔNG truyền sentinel — đây là sự khác biệt chính so với AdaptiveAttention
            context, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
                pm,
            )                                                        # [B_t, enc_dim], [B_t, N]

            # LSTM step: input = [word_embedding || attention_context]
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], context], dim=1),
                (h[:batch_size_t], c[:batch_size_t]),
            )                                                        # [B_t, decoder_dim]

            preds = self.fc(self.dropout(h))                        # [B_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :]      = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
