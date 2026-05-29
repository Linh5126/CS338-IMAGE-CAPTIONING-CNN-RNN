import torch
from torch import nn
import torchvision
from ultralytics import YOLO
import os
import torchvision.ops as ops

TOP_K_OBJECTS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Fusion 5D
# ═══════════════════════════════════════════════════════════════════════════════
class SpatialFusion5D(nn.Module):
    """
    Encode bounding box [cx/W, cy/H, w/W, h/H, area] → encoder_dim.
    Cộng additive vào object feature sau ROI Align.
    Chiều thứ 5 (diện tích) giúp phân biệt foreground vs background —
    quan trọng với Flickr8k ("a dog in the background", "two people in front").
    """

    def __init__(self, encoder_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(5, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, encoder_dim),
        )
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _boxes_to_5d(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx   = (x1 + x2) / 2.0 / W
        cy   = (y1 + y2) / 2.0 / H
        bw   = (x2 - x1).clamp(min=0) / W
        bh   = (y2 - y1).clamp(min=0) / H
        area = bw * bh
        return torch.stack([cx, cy, bw, bh, area], dim=1)

    def forward(self, feat: torch.Tensor, boxes: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        if feat.size(0) == 0:
            return feat
        return feat + self.proj(self._boxes_to_5d(boxes, H, W))


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-Stream Encoder
# ═══════════════════════════════════════════════════════════════════════════════
class DualStreamEncoder(nn.Module):
    def __init__(self, encoded_image_size: int = 14, yolo_version: str = 'yolo12s.pt'):
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.yolo_version   = yolo_version

        resnet             = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT
        )
        self.resnet        = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        yolo_instance  = YOLO(self.yolo_version)
        self.yolo_core = yolo_instance.model
        for p in self.yolo_core.parameters():
            p.requires_grad = False
        self._yolo_wrapper = [yolo_instance]

        yolo_feat_ch = self._probe_yolo_channels(yolo_instance)
        print(f"[DualStreamEncoder] YOLO feature channels: {yolo_feat_ch}")

        self.yolo_proj = nn.Linear(yolo_feat_ch, 2048)
        nn.init.xavier_uniform_(self.yolo_proj.weight)
        nn.init.zeros_(self.yolo_proj.bias)

        self.spatial_fusion  = SpatialFusion5D(encoder_dim=2048)
        self.global_type_emb = nn.Parameter(torch.zeros(1, 1, 2048))
        self.local_type_emb  = nn.Parameter(torch.zeros(1, 1, 2048))
        nn.init.normal_(self.global_type_emb, std=0.01)
        nn.init.normal_(self.local_type_emb,  std=0.01)

    @staticmethod
    def _probe_yolo_channels(yolo_instance) -> int:
        feats  = []
        handle = yolo_instance.model.model[-2].register_forward_hook(
            lambda m, inp, o: feats.append(o)
        )
        try:
            with torch.no_grad():
                yolo_instance.model(torch.zeros(1, 3, 256, 256))
        finally:
            handle.remove()
        f = feats[0]
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
        yolo_instance = _YOLO(os.environ.get('YOLO_WEIGHTS', getattr(self, 'yolo_version', 'yolo12s.pt')))
        yolo_instance.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.yolo_core = yolo_instance.model
        for p in self.yolo_core.parameters():
            p.requires_grad = False
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
                p = p[p[:, 4] > conf_thresh]
                if len(p) == 0:
                    continue
                p  = p[p[:, 4].argsort(descending=True)]
                cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                results[i] = torch.stack([
                    (cx - w / 2).clamp(0, W), (cy - h / 2).clamp(0, H),
                    (cx + w / 2).clamp(0, W), (cy + h / 2).clamp(0, H),
                ], dim=1).to(device)
        except Exception as e:
            print(f"[WARN] _parse_yolo12_boxes_batched: {e}")
        return results

    def forward(self, images: torch.Tensor):
        device     = images.device
        batch_size = images.size(0)
        H, W       = images.shape[-2], images.shape[-1]

        gf = self.resnet(images)
        gf = self.adaptive_pool(gf)
        gf = gf.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2048)
        gf = gf + self.global_type_emb

        mean        = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std         = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        yolo_images = (images * std + mean).clamp(0.0, 1.0)

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
        f_map         = f_map.detach().float()
        spatial_scale = float(f_map.shape[-1]) / W

        boxes_per_img   = self._parse_yolo12_boxes_batched(
            raw_output, batch_size, (H, W), device
        )
        roi_list        = []
        real_boxes_list = []
        n_per_img       = []
        for boxes in boxes_per_img:
            b = boxes[:TOP_K_OBJECTS] if (boxes is not None and len(boxes) > 0) \
                else torch.zeros(0, 4, device=device)
            roi_list.append(b)
            real_boxes_list.append(b)
            n_per_img.append(len(b))

        obj_vecs   = torch.zeros(batch_size, TOP_K_OBJECTS, 2048, device=device)
        is_padding = torch.ones(batch_size, TOP_K_OBJECTS, dtype=torch.bool, device=device)

        if any(n > 0 for n in n_per_img):
            roi_out = ops.roi_align(
                f_map, roi_list, output_size=(1, 1), spatial_scale=spatial_scale,
            )
            idx = 0
            for i, n in enumerate(n_per_img):
                if n > 0:
                    vecs = roi_out[idx : idx + n].view(n, -1)
                    vecs = self.yolo_proj(vecs)
                    vecs = self.spatial_fusion(vecs, real_boxes_list[i], H, W)
                    obj_vecs[i, :n]   = vecs
                    is_padding[i, :n] = False
                idx += n

        real = ~is_padding
        if real.any():
            obj_vecs[real] = obj_vecs[real] + self.local_type_emb.view(2048)

        combined      = torch.cat([gf, obj_vecs], dim=1)
        global_pad    = torch.zeros(batch_size, gf.size(1), dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([global_pad, is_padding], dim=1)
        return combined, full_pad_mask

    def fine_tune(self, fine_tune: bool = True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.yolo_proj.parameters():
            p.requires_grad = True
        for p in self.spatial_fusion.parameters():
            p.requires_grad = True
        if fine_tune:
            for child in list(self.resnet.children())[5:]:
                for p in child.parameters():
                    p.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Attention
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.sentinel_proj = nn.Linear(decoder_dim,  encoder_dim)
        self.encoder_att   = nn.Linear(encoder_dim,  attention_dim)
        self.decoder_att   = nn.Linear(decoder_dim,  attention_dim)
        self.full_att      = nn.Linear(attention_dim, 1)
        self.relu          = nn.ReLU()
        self.softmax       = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, sentinel, padding_mask=None):
        s_proj = self.sentinel_proj(sentinel)
        full   = torch.cat([encoder_out, s_proj.unsqueeze(1)], dim=1)
        att1   = self.encoder_att(full)
        att2   = self.decoder_att(decoder_hidden)
        att    = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        if padding_mask is not None:
            sent_mask = torch.zeros(att.size(0), 1, dtype=torch.bool, device=att.device)
            att = att.masked_fill(
                torch.cat([padding_mask, sent_mask], dim=1), float('-inf')
            )
        alpha   = self.softmax(att)
        context = (full * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder LSTM — hỗ trợ 3 chế độ:
#   forward()         — teacher forcing (XE training)
#   forward_sample()  — stochastic sampling  (SCST: lấy mẫu để tính reward)
#   forward_greedy()  — greedy decode        (SCST: baseline reward)
# ═══════════════════════════════════════════════════════════════════════════════
class DecoderAdaptive(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.decoder_dim   = decoder_dim
        self.vocab_size    = vocab_size
        self.dropout_p     = dropout

        self.attention    = AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.dropout      = nn.Dropout(p=dropout)
        self.decode_step  = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h       = nn.Linear(encoder_dim, decoder_dim)
        self.init_c       = nn.Linear(encoder_dim, decoder_dim)
        self.sentinel_w_x = nn.Linear(embed_dim,   decoder_dim)
        self.sentinel_w_h = nn.Linear(decoder_dim, decoder_dim)
        self.fc           = nn.Linear(decoder_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    # ── Teacher Forcing (XE) ──────────────────────────────────────────────────
    def forward(self, encoder_out, encoded_captions, caption_lengths,
                padding_mask=None):
        batch_size = encoder_out.size(0)
        num_slots  = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        if padding_mask is not None:
            padding_mask = padding_mask[sort_ind]

        embeddings     = self.embedding(encoded_captions)
        h, c           = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        max_t          = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_t, self.vocab_size,
                                  device=encoder_out.device)
        alphas      = torch.zeros(batch_size, max_t, num_slots + 1,
                                  device=encoder_out.device)

        dl_t          = torch.tensor(decode_lengths, device=encoder_out.device)
        t_range       = torch.arange(max_t,          device=encoder_out.device)
        batch_sizes_t = (dl_t.unsqueeze(1) > t_range.unsqueeze(0)).sum(0).tolist()

        for t in range(max_t):
            bst = batch_sizes_t[t]
            pm  = padding_mask[:bst] if padding_mask is not None else None
            g_t = torch.sigmoid(
                self.sentinel_w_x(embeddings[:bst, t, :])
                + self.sentinel_w_h(h[:bst])
            )
            s_t            = g_t * torch.tanh(c[:bst])
            context, alpha = self.attention(encoder_out[:bst], h[:bst], s_t, pm)
            h, c           = self.decode_step(
                torch.cat([embeddings[:bst, t, :], context], dim=1),
                (h[:bst], c[:bst]),
            )
            predictions[:bst, t, :] = self.fc(self.dropout(h))
            alphas[:bst, t, :]      = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    # ── Shared decode step (dùng nội bộ cho sample/greedy) ───────────────────
    def _decode_one_step(self, encoder_out, h, c, prev_token_emb, padding_mask):
        g_t = torch.sigmoid(
            self.sentinel_w_x(prev_token_emb) + self.sentinel_w_h(h)
        )
        s_t            = g_t * torch.tanh(c)
        context, alpha = self.attention(encoder_out, h, s_t, padding_mask)
        h, c           = self.decode_step(
            torch.cat([prev_token_emb, context], dim=1), (h, c)
        )
        logits = self.fc(h)    # không dropout trong inference
        return logits, h, c

    # ── Stochastic Sampling (SCST: lấy mẫu để tính reward) ──────────────────
    def forward_sample(self, encoder_out, start_token: int, end_token: int,
                       max_len: int = 50, padding_mask=None):
        """
        Sinh câu bằng categorical sampling — dùng để tính reward trong SCST.

        Returns:
            sampled_ids : [B, max_len]   int64 — token ids được sample
            log_probs   : [B, max_len]   float — log P(token | context)
            seq_lengths : [B]            int   — độ dài thực (kể cả <end> nếu có)
        """
        batch_size  = encoder_out.size(0)
        device      = encoder_out.device
        h, c        = self.init_hidden_state(encoder_out)

        # Bắt đầu bằng <start>
        prev_word = torch.full((batch_size,), start_token, dtype=torch.long, device=device)

        sampled_ids = torch.zeros(batch_size, max_len, dtype=torch.long,  device=device)
        log_probs   = torch.zeros(batch_size, max_len, dtype=torch.float, device=device)
        # Theo dõi những sequence chưa sinh ra <end>
        active      = torch.ones(batch_size, dtype=torch.bool, device=device)
        seq_lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=device)

        for t in range(max_len):
            if not active.any():
                break
            emb    = self.embedding(prev_word)                   # [B, embed_dim]
            logits, h, c = self._decode_one_step(
                encoder_out, h, c, emb, padding_mask
            )
            # Categorical sampling
            probs     = torch.softmax(logits, dim=1)             # [B, vocab]
            sampled   = torch.multinomial(probs, 1).squeeze(1)   # [B]
            lp        = torch.log(probs.gather(1, sampled.unsqueeze(1)).squeeze(1) + 1e-10)

            sampled_ids[:, t] = sampled
            log_probs[:, t]   = lp

            # Đánh dấu sequence kết thúc
            just_ended        = active & (sampled == end_token)
            seq_lengths[just_ended & (seq_lengths == max_len)] = t + 1
            active            = active & (sampled != end_token)

            prev_word = sampled

        # Sequence không sinh ra <end> trước max_len → giữ max_len
        return sampled_ids, log_probs, seq_lengths

    # ── Greedy Decode (SCST: baseline) ────────────────────────────────────────
    def forward_greedy(self, encoder_out, start_token: int, end_token: int,
                       max_len: int = 50, padding_mask=None):
        """
        Greedy decode — baseline reward trong SCST.
        Không cần gradient.

        Returns:
            greedy_ids  : [B, max_len]   int64
            seq_lengths : [B]            int
        """
        batch_size  = encoder_out.size(0)
        device      = encoder_out.device
        h, c        = self.init_hidden_state(encoder_out)
        prev_word   = torch.full((batch_size,), start_token, dtype=torch.long, device=device)

        greedy_ids  = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        active      = torch.ones(batch_size,  dtype=torch.bool, device=device)
        seq_lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(max_len):
                if not active.any():
                    break
                emb           = self.embedding(prev_word)
                logits, h, c  = self._decode_one_step(
                    encoder_out, h, c, emb, padding_mask
                )
                chosen = logits.argmax(dim=1)                    # [B]
                greedy_ids[:, t] = chosen

                just_ended  = active & (chosen == end_token)
                seq_lengths[just_ended & (seq_lengths == max_len)] = t + 1
                active      = active & (chosen != end_token)
                prev_word   = chosen

        return greedy_ids, seq_lengths
