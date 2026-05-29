import torch
from torch import nn
import torchvision
from ultralytics import YOLO
import torchvision.ops as ops

# Số object slots cố định — tránh biến động giữa các batch
TOP_K_OBJECTS = 10


# ── Dual-Stream Encoder ────────────────────────────────────────────────────────
class DualStreamEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, yolo_version='yolo12s.pt'):
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.yolo_version   = yolo_version

        # ── ResNet101 Backbone (Global Features) ──────────────────────────────
        resnet = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT
        )
        modules      = list(resnet.children())[:-2]   # giữ conv → layer4, bỏ avgpool+fc
        self.resnet  = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # ── YOLO Backbone (Local / Object Features) ───────────────────────────
        yolo_instance    = YOLO(self.yolo_version)
        self.yolo_core   = yolo_instance.model
        for param in self.yolo_core.parameters():
            param.requires_grad = False

        # Lưu yolo_instance ngoài nn.Module để không bị serialize cùng state_dict
        self._yolo_wrapper = [yolo_instance]

        # Probe kênh feature map của YOLO một lần tại __init__
        yolo_feat_channels = self._probe_yolo_channels(yolo_instance)
        print(f"[DualStreamEncoder] YOLO feature channels detected: {yolo_feat_channels}")

        # Chiếu YOLO feature lên 2048-d để khớp ResNet
        self.yolo_proj = nn.Linear(yolo_feat_channels, 2048)
        nn.init.xavier_uniform_(self.yolo_proj.weight)
        nn.init.zeros_(self.yolo_proj.bias)

        # Type embeddings: phân biệt global vs local cho decoder
        self.global_type_emb = nn.Parameter(torch.zeros(1, 1, 2048))
        self.local_type_emb  = nn.Parameter(torch.zeros(1, 1, 2048))
        nn.init.normal_(self.global_type_emb, std=0.01)
        nn.init.normal_(self.local_type_emb,  std=0.01)

    # ── Probe channel dim của YOLO feature map ────────────────────────────────
    @staticmethod
    def _probe_yolo_channels(yolo_instance):
        """Chạy một forward dummy để lấy số kênh của layer[-2]."""
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

    # ── Serialization helpers ─────────────────────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_yolo_wrapper', None)
        if '_modules' in state:
            state['_modules'] = state['_modules'].copy()
            state['_modules'].pop('yolo_core', None)
        return state

    def __setstate__(self, state):
        # FIX: gọi super().__setstate__ để đảm bảo nn.Module khởi tạo đầy đủ
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

    # ── Parse YOLO boxes cho toàn bộ batch (batched, thay vì per-image loop) ──
    def _parse_yolo12_boxes_batched(self, raw_output, batch_size, img_shape,
                                    device, conf_thresh=0.25):
        """
        Parse bounding boxes từ raw output YOLO (batched inference).

        Trả về list[Tensor | None] độ dài batch_size.
        Mỗi phần tử là [n, 4] xyxy hoặc None nếu không có box nào hợp lệ.
        """
        H, W    = img_shape
        results = [None] * batch_size

        try:
            # ── Chuẩn hoá output về tensor preds ─────────────────────────────
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

            # ── Đảm bảo shape [B, N, 5+] ─────────────────────────────────────
            if preds.dim() == 2:
                preds = preds.unsqueeze(0)          # [N, 5+] → [1, N, 5+]

            if preds.dim() != 3:
                return results

            # Transpose nếu shape là [B, 5+, N]
            if preds.shape[1] <= 6 and preds.shape[2] > 6:
                preds = preds.permute(0, 2, 1).contiguous()

            # ── Parse từng ảnh trong batch ────────────────────────────────────
            for i in range(min(preds.shape[0], batch_size)):
                p = preds[i]                             # [N, 5+]
                if p.shape[-1] < 5:
                    continue
                mask = p[:, 4] > conf_thresh
                p    = p[mask]
                if len(p) == 0:
                    continue

                # Sort theo confidence giảm dần → top-K sau sẽ là tốt nhất
                p = p[p[:, 4].argsort(descending=True)]

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

        return results   # trả về list, không bao giờ trả về None đơn lẻ

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, images):
        """
        FIX HIỆU NĂNG: YOLO chỉ inference MỘT LẦN cho toàn batch (thay vì loop per-image).
        ROI Align cũng được batched trong một lệnh.

        Returns:
            combined      : [B, 196 + TOP_K, 2048]  — global + local features
            full_pad_mask : [B, 196 + TOP_K]  bool, True = slot là padding
        """
        device     = images.device
        batch_size = images.size(0)

        # ── 1. Global features từ ResNet101 ───────────────────────────────────
        global_features = self.resnet(images)
        global_features = self.adaptive_pool(global_features)
        global_features = global_features.permute(0, 2, 3, 1).contiguous()
        global_features = global_features.view(batch_size, -1, 2048)
        global_features = global_features + self.global_type_emb     # [B, 196, 2048]

        # ── 2. De-normalize ảnh về [0,1] cho YOLO ────────────────────────────
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        yolo_images = torch.clamp(images * std + mean, 0.0, 1.0)

        # ── 3. BATCHED YOLO inference — một hook, một forward call ────────────
        #   Trước đây: loop batch_size lần → N hook registrations, N inference calls
        #   Bây giờ:   1 hook registration, 1 inference call → nhanh hơn ~10-30x
        hook_feats = []
        handle = self.yolo_core.model[-2].register_forward_hook(
            lambda m, inp, o: hook_feats.append(o)
        )
        with torch.no_grad():
            raw_output = self.yolo_core(yolo_images)
        handle.remove()

        # Feature map của YOLO: [B, C, H', W']
        f_map = hook_feats[0]
        if isinstance(f_map, (list, tuple)):
            f_map = f_map[1] if len(f_map) > 1 else f_map[0]
        # detach + float32 vì YOLO frozen; roi_align cần float32
        f_map = f_map.detach().float()

        H, W          = yolo_images.shape[-2:]
        spatial_scale = float(f_map.shape[-1]) / W

        # ── 4. Parse boxes cho tất cả ảnh ────────────────────────────────────
        boxes_per_img = self._parse_yolo12_boxes_batched(
            raw_output, batch_size, (H, W), device
        )

        # Chuẩn bị danh sách boxes cho batched ROI Align
        roi_list   = []
        n_per_img  = []
        for boxes in boxes_per_img:
            if boxes is not None and len(boxes) > 0:
                b = boxes[:TOP_K_OBJECTS]
            else:
                b = torch.zeros(0, 4, device=device)
            roi_list.append(b)
            n_per_img.append(len(b))

        # ── 5. Batched ROI Align ──────────────────────────────────────────────
        obj_vecs   = torch.zeros(batch_size, TOP_K_OBJECTS, 2048, device=device)
        is_padding = torch.ones(batch_size,  TOP_K_OBJECTS, dtype=torch.bool, device=device)

        if any(n > 0 for n in n_per_img):
            # ops.roi_align nhận list[Tensor] → xử lý batch trong một lệnh
            roi_out = ops.roi_align(
                f_map, roi_list,
                output_size=(1, 1),
                spatial_scale=spatial_scale,
            )                                                    # [sum(n_i), C, 1, 1]

            idx = 0
            for i, n in enumerate(n_per_img):
                if n > 0:
                    vecs = roi_out[idx : idx + n].view(n, -1)   # [n, C]
                    vecs = self.yolo_proj(vecs)                  # [n, 2048] — có gradient
                    obj_vecs[i, :n]   = vecs
                    is_padding[i, :n] = False
                idx += n

        # ── 6. Type embedding cho các slot thực ──────────────────────────────
        real = ~is_padding                                       # [B, TOP_K]
        if real.any():
            obj_vecs[real] = obj_vecs[real] + self.local_type_emb.view(2048)

        # ── 7. Ghép global + local ────────────────────────────────────────────
        combined      = torch.cat([global_features, obj_vecs], dim=1)  # [B, 206, 2048]
        global_pad    = torch.zeros(
            batch_size, global_features.size(1), dtype=torch.bool, device=device
        )
        full_pad_mask = torch.cat([global_pad, is_padding], dim=1)     # [B, 206]

        return combined, full_pad_mask

    def fine_tune(self, fine_tune=True):
        """
        Đóng băng toàn bộ ResNet101 theo mặc định.
        Nếu fine_tune=True: mở thêm layer2, layer3, layer4 (index 5, 6, 7).
        yolo_proj và type embeddings luôn trainable.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        for p in self.yolo_proj.parameters():
            p.requires_grad = True
        # global_type_emb và local_type_emb là nn.Parameter — luôn requires_grad=True

        if fine_tune:
            # FIX comment: mở layer2 (idx5), layer3 (idx6), layer4 (idx7)
            for child in list(self.resnet.children())[5:]:
                for p in child.parameters():
                    p.requires_grad = True


# ── Adaptive Attention với Sentinel + Padding Mask ────────────────────────────
class AdaptiveAttention(nn.Module):
    """
    Sentinel-based adaptive attention (Lu et al., 2017).
    Thêm padding_mask để loại bỏ các local object slot không có thật.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.sentinel_proj = nn.Linear(decoder_dim, encoder_dim)
        self.encoder_att   = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att   = nn.Linear(decoder_dim, attention_dim)
        self.full_att      = nn.Linear(attention_dim, 1)
        self.relu          = nn.ReLU()
        self.softmax       = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, sentinel, padding_mask=None):
        """
        Args:
            encoder_out   : [B, N, encoder_dim]
            decoder_hidden: [B, decoder_dim]
            sentinel      : [B, decoder_dim]
            padding_mask  : [B, N] bool — True = slot là padding (masked thành -inf)
        Returns:
            attention_weighted_encoding: [B, encoder_dim]
            alpha                      : [B, N+1]
        """
        sentinel_projected = self.sentinel_proj(sentinel)            # [B, 2048]

        combined_out = torch.cat(
            [encoder_out, sentinel_projected.unsqueeze(1)], dim=1
        )                                                            # [B, N+1, 2048]

        att1 = self.encoder_att(combined_out)                        # [B, N+1, att_dim]
        att2 = self.decoder_att(decoder_hidden)                      # [B, att_dim]
        att  = self.full_att(
            self.relu(att1 + att2.unsqueeze(1))
        ).squeeze(2)                                                 # [B, N+1]

        # Mask padding slots trước khi softmax
        if padding_mask is not None:
            sentinel_mask = torch.zeros(
                att.size(0), 1, dtype=torch.bool, device=att.device
            )
            full_mask = torch.cat([padding_mask, sentinel_mask], dim=1)  # [B, N+1]
            att = att.masked_fill(full_mask, float('-inf'))

        alpha                       = self.softmax(att)              # [B, N+1]
        attention_weighted_encoding = (
            combined_out * alpha.unsqueeze(2)
        ).sum(dim=1)                                                 # [B, 2048]

        return attention_weighted_encoding, alpha


# ── LSTM Decoder với Adaptive Attention ───────────────────────────────────────
class DecoderAdaptive(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.decoder_dim   = decoder_dim
        self.vocab_size    = vocab_size
        self.dropout_p     = dropout

        self.attention = AdaptiveAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout   = nn.Dropout(p=dropout)

        # LSTMCell input = word_embed (512) + context (2048) = 2560
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.init_h        = nn.Linear(encoder_dim, decoder_dim)
        self.init_c        = nn.Linear(encoder_dim, decoder_dim)
        self.sentinel_w_x  = nn.Linear(embed_dim,   decoder_dim)
        self.sentinel_w_h  = nn.Linear(decoder_dim, decoder_dim)
        self.fc            = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, padding_mask=None):
        """
        Args:
            encoder_out      : [B, N, encoder_dim]
            encoded_captions : [B, max_len]
            caption_lengths  : [B, 1]
            padding_mask     : [B, N] bool
        Returns:
            predictions  : [B, max_decode_len, vocab_size]
            caps_sorted  : [B, max_len]
            decode_lengths: list[int]
            alphas       : [B, max_decode_len, N+1]
            sort_ind     : sort indices
        """
        batch_size      = encoder_out.size(0)
        vocab_size      = self.vocab_size
        num_slots       = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        if padding_mask is not None:
            padding_mask = padding_mask[sort_ind]

        embeddings     = self.embedding(encoded_captions)      # [B, max_len, embed_dim]
        h, c           = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        max_t          = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_t, vocab_size,    device=encoder_out.device)
        alphas      = torch.zeros(batch_size, max_t, num_slots + 1, device=encoder_out.device)

        # FIX HIỆU NĂNG: precompute batch_size_t cho mọi timestep (vectorized, một lần)
        dl_t          = torch.tensor(decode_lengths, device=encoder_out.device)
        t_range       = torch.arange(max_t,          device=encoder_out.device)
        batch_sizes_t = (dl_t.unsqueeze(1) > t_range.unsqueeze(0)).sum(0).tolist()

        for t in range(max_t):
            batch_size_t = batch_sizes_t[t]

            pm = padding_mask[:batch_size_t] if padding_mask is not None else None

            # Sentinel vector s_t
            g_t = torch.sigmoid(
                self.sentinel_w_x(embeddings[:batch_size_t, t, :])
                + self.sentinel_w_h(h[:batch_size_t])
            )
            s_t = g_t * torch.tanh(c[:batch_size_t])               # [B_t, decoder_dim]

            # Adaptive Attention
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t], s_t, pm
            )

            # LSTM step
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]),
            )

            preds = self.fc(self.dropout(h))                        # [B_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :]      = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
