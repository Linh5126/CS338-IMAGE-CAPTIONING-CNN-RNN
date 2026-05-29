from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from demo_engine import load_model, run_demo_inference
from demo_metrics import score_single_image
from visualization import combined_attention_map, overlay_heatmap, draw_boxes, attention_breakdown
from sota_compare import (
    SOTA_MODELS,
    load_sota_model,
    generate_sota_caption,
    ours_comparison_row,
    rows_for_display,
)


st.set_page_config(
    page_title="Image Captioning Demo · Dual Encoder",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --card: #ffffff; --ink: #101828; --muted: #475467; --line: #d0d5dd; }
.main-title {font-size: 2.1rem; font-weight: 760; letter-spacing: -0.02em; margin-bottom: 0.2rem; color: #f8fafc;}
.subtle {color: #cbd5e1; font-size: 0.98rem;}
.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.0rem 1.1rem;
    margin-bottom: 0.8rem;
    color: var(--ink) !important;
}
.card * { color: var(--ink) !important; }
.caption-box {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    font-size: 1.25rem;
    font-weight: 650;
    color: #101828 !important;
}
.metric-note {font-size: 0.86rem; color: #98a2b3;}
.small-code {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.85rem; color: #344054 !important;}
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "checkpoints" / "BEST_checkpoint_dual_stream_lstm_adaptive_FS5_RL_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar"
DEFAULT_WMAP = ROOT / "checkpoints" / "WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json"
DEFAULT_YOLO = ROOT / "checkpoints" / "yolo12s.pt"


@st.cache_resource(show_spinner=False)
def cached_load_model(checkpoint_path, word_map_path, device, yolo_weights):
    return load_model(checkpoint_path, word_map_path, device=device, yolo_weights=yolo_weights or None)


@st.cache_resource(show_spinner=False)
def cached_sota_model(model_key, device, dtype):
    return load_sota_model(model_key, device=device, torch_dtype=dtype)


with st.sidebar:
    st.markdown("### Cấu hình mô hình")
    checkpoint_path = st.text_input("Checkpoint (.pth.tar)", str(DEFAULT_CKPT))
    word_map_path = st.text_input("Word map (.json)", str(DEFAULT_WMAP))
    yolo_weights = st.text_input("YOLO weights (.pt, nếu cần)", str(DEFAULT_YOLO) if DEFAULT_YOLO.exists() else "")
    device = st.selectbox("Thiết bị", ["auto", "cuda", "cpu"], index=0)

    st.markdown("### Giải mã")
    decode_mode = st.selectbox("Chế độ", ["both", "greedy", "beam"], index=0, help="Heatmap được bám theo đường Greedy để dễ giải thích từng từ.")
    beam_size = st.slider("Beam size", 2, 10, 5)
    max_len = st.slider("Độ dài tối đa", 10, 80, 50)
    lp_alpha = st.slider("Length penalty α", 0.0, 1.5, 0.7, 0.1)

    st.markdown("### Hiển thị")
    heat_alpha = st.slider("Độ đậm heatmap", 0.15, 0.85, 0.48, 0.05)
    global_weight = st.slider("Tỷ trọng heatmap toàn cục", 0.0, 1.0, 0.65, 0.05)

    st.markdown("### So sánh mô hình hiện đại")
    enable_sota = st.checkbox("Bật tab so sánh BLIP/GIT/BLIP-2", value=False)
    sota_labels = {k: f"{v.display_name} — {v.hf_id}" for k, v in SOTA_MODELS.items()}
    sota_selected = st.multiselect(
        "Chọn mô hình so sánh",
        options=list(SOTA_MODELS.keys()),
        default=["blip_base", "git_base_coco"],
        format_func=lambda k: sota_labels[k],
        help="BLIP-2 rất nặng; nên thử BLIP-base/GIT-base trước."
    )
    sota_dtype = st.selectbox("Kiểu số cho baseline", ["auto", "float16", "bfloat16"], index=0)
    sota_tokens = st.slider("Max new tokens baseline", 10, 80, 40)

st.markdown('<div class="main-title">Demo Image Captioning · Dual Encoder + Adaptive Attention</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Thả ảnh vào để sinh caption, xem vùng chú ý theo từng từ, kiểm tra các vật thể từ YOLO và tính metrics nếu có caption tham chiếu.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Kéo thả một ảnh vào đây", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Đặt checkpoint, word map và YOLO weights vào thư mục `checkpoints/`, sau đó tải ảnh để chạy demo.")
    st.markdown(
        """
**Luồng demo sẽ hiển thị:**
1. Ảnh đầu vào sau tiền xử lý 256×256.  
2. Caption Greedy và Beam Search.  
3. Heatmap chú ý theo từng từ trong caption.  
4. Hộp vật thể YOLO và trọng số chú ý object/global/sentinel.  
5. Metrics nếu bạn nhập caption tham chiếu.
"""
    )
    st.stop()

try:
    with st.spinner("Đang tải checkpoint và mô hình..."):
        loaded = cached_load_model(checkpoint_path, word_map_path, device, yolo_weights)
except Exception as e:
    st.error("Không tải được mô hình. Kiểm tra lại checkpoint, WORDMAP và YOLO weights.")
    st.exception(e)
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")

try:
    with st.spinner("Đang chạy encoder, decoder và thu thập attention trace..."):
        result = run_demo_inference(
            loaded,
            pil_img,
            decode_mode=decode_mode,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty_alpha=lp_alpha,
        )
except Exception as e:
    st.error("Lỗi khi suy luận. Nếu lỗi liên quan YOLO, hãy chắc chắn file yolo12s.pt có sẵn trong checkpoints/ hoặc truyền đúng đường dẫn.")
    st.exception(e)
    st.stop()

image = result["image"]
trace = result["encoder_trace"]
greedy = result["greedy"]
beam = result.get("beam")

left, right = st.columns([0.92, 1.08], gap="large")
with left:
    st.image(image, caption="Ảnh đầu vào sau khi resize về 256×256", use_container_width=True)
    st.markdown("#### Caption")
    st.markdown(f'<div class="caption-box">{greedy["caption"] or "<rỗng>"}</div>', unsafe_allow_html=True)
    st.caption(f"Greedy · độ dài {len(greedy['words'])} từ · xác suất token trung bình {greedy['avg_token_prob']:.3f}")
    if beam is not None:
        st.markdown(f'<div class="caption-box">{beam["caption"] or "<rỗng>"}</div>', unsafe_allow_html=True)
        st.caption(f"Beam{beam_size} · score chuẩn hóa {beam['score']:.3f}")

with right:
    st.markdown("#### Tóm tắt vận hành")
    t = result["timing"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global slots", trace["global_slots"])
    c2.metric("Object slots", trace["object_slots"], f"thật: {trace['valid_objects'][0]}")
    c3.metric("Tổng slots", result["encoder_out"].shape[1])
    c4.metric("Thời gian", f"{t['total_ms']:.0f} ms")

    box_source = trace.get("box_source", "unknown")
    source_note = "hậu xử lý Ultralytics" if box_source == "ultralytics_postprocess" else "fallback đọc tensor thô"
    st.markdown(
        f"""
<div class="card">
<b>Encoder output</b>: <span class="small-code">{trace['encoder_out_shape']}</span><br/>
<b>Global stream</b>: ResNet feature map <span class="small-code">{trace['global_feature_map_shape']}</span>, grid <span class="small-code">{trace['global_grid'][0]}×{trace['global_grid'][1]}</span><br/>
<b>YOLO stream</b>: feature map <span class="small-code">{trace['yolo_feature_map_shape']}</span>, lấy tối đa <span class="small-code">{trace['object_slots']}</span> vùng vật thể<br/>
<b>Nguồn hộp YOLO</b>: <span class="small-code">{source_note}</span><br/>
<b>Decoder</b>: mỗi bước sinh từ lấy attention trên global slots + object slots + sentinel.
</div>
""",
        unsafe_allow_html=True,
    )
    if box_source != "ultralytics_postprocess":
        st.warning("Đang dùng fallback đọc tensor thô của YOLO. Nếu hộp bị dồn về góc trên trái, hãy kiểm tra file yolo12s.pt/phiên bản ultralytics hoặc dùng bản demo v3.")

steps = greedy["steps"]
if not steps:
    st.warning("Không có bước giải mã nào để hiển thị heatmap.")
    st.stop()

st.divider()

tabs = st.tabs(["Heatmap theo từng từ", "Vật thể YOLO", "Bảng bước giải mã", "Đánh giá", "Nhật ký pipeline", "So sánh SOTA"])

with tabs[0]:
    st.markdown("### Attention heatmap")
    words_for_slider = [f"{i+1}. {s['word']}" for i, s in enumerate(steps)]
    idx = st.select_slider("Chọn từ trong quá trình sinh caption", options=list(range(len(steps))), format_func=lambda i: words_for_slider[i])
    step = steps[idx]
    maps = combined_attention_map(step["alpha"], trace, image.size, global_weight=global_weight)
    overlay = overlay_heatmap(image, maps["combined"], alpha=heat_alpha)

    a, b = st.columns([1.0, 1.0], gap="large")
    with a:
        st.image(overlay, caption=f"Vùng chú ý khi sinh từ: {step['word']}", use_container_width=True)
    with b:
        breakdown = attention_breakdown(step["alpha"], trace)
        st.markdown("#### Tỷ lệ chú ý")
        st.progress(min(1.0, breakdown["global"]), text=f"Toàn cục ResNet: {breakdown['global']:.3f}")
        st.progress(min(1.0, breakdown["object"]), text=f"Vật thể YOLO: {breakdown['object']:.3f}")
        st.progress(min(1.0, breakdown["sentinel"]), text=f"Sentinel/ngôn ngữ: {breakdown['sentinel']:.3f}")
        st.markdown("#### Các từ ứng viên cao nhất")
        st.dataframe(pd.DataFrame(step["top_alternatives"]), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.image(overlay_heatmap(image, maps["global"], alpha=heat_alpha), caption="Chỉ global slots", use_container_width=True)
    col2.image(overlay_heatmap(image, maps["object"], alpha=heat_alpha), caption="Chỉ object slots", use_container_width=True)
    col3.image(overlay, caption="Kết hợp", use_container_width=True)

with tabs[1]:
    st.markdown("### Vùng vật thể do YOLO cung cấp cho captioning")
    boxes = trace["boxes"][0] if trace.get("boxes") else []
    if boxes:
        # Use attention at selected word for object scores.
        g = int(trace["global_slots"])
        scores = step["alpha"][g:g+len(boxes)]
        st.image(draw_boxes(image, boxes, scores=scores), caption="Hộp vật thể + trọng số chú ý tại từ đang chọn", use_container_width=True)
        labels = trace.get("box_labels", [[]])[0] if trace.get("box_labels") else []
        confs = trace.get("box_confidences", [[]])[0] if trace.get("box_confidences") else []
        obj_df = pd.DataFrame([
            {
                "object_slot": i + 1,
                "label": labels[i] if i < len(labels) else "object",
                "conf": round(float(confs[i]), 4) if i < len(confs) else None,
                "x1": round(b[0], 1), "y1": round(b[1], 1), "x2": round(b[2], 1), "y2": round(b[3], 1),
                "attention": round(float(scores[i]), 5) if i < len(scores) else 0.0,
            }
            for i, b in enumerate(boxes)
        ])
        st.dataframe(obj_df, use_container_width=True, hide_index=True)
    else:
        st.warning("YOLO không phát hiện được vật thể hợp lệ trong ảnh này hoặc parser output chưa khớp với phiên bản YOLO đang dùng.")

with tabs[2]:
    st.markdown("### Bảng bước giải mã")
    rows = []
    for s in steps:
        br = attention_breakdown(s["alpha"], trace)
        rows.append({
            "t": s["t"],
            "word": s["word"],
            "prob": round(s["prob"], 5),
            "global_att": round(br["global"], 5),
            "object_att": round(br["object"], 5),
            "sentinel_att": round(br["sentinel"], 5),
            "context_norm": round(s["context_norm"], 3),
            "gate_mean": round(s["gate_mean"], 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown("### Đánh giá caption với caption tham chiếu")
    st.caption("Nhập mỗi caption tham chiếu trên một dòng. Với ảnh ngoài Flickr8k, phần này có thể bỏ trống.")
    refs_text = st.text_area("Caption tham chiếu", height=150, placeholder="a dog is running through the grass\na brown dog runs in a field")
    target_caption = st.radio("Caption cần đánh giá", ["Greedy", f"Beam{beam_size}"], horizontal=True)
    if st.button("Tính metrics cho ảnh này"):
        hyp = greedy["caption"] if target_caption == "Greedy" or beam is None else beam["caption"]
        refs = [r for r in refs_text.splitlines() if r.strip()]
        scores = score_single_image(hyp, refs)
        st.json(scores)
        st.markdown('<div class="metric-note">Lưu ý: metrics một ảnh chỉ dùng để demo/giải thích. Báo cáo chính nên dùng eval.py trên toàn bộ TEST set.</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown("### Nhật ký pipeline")
    pipeline = [
        {"bước": 1, "khối": "Tiền xử lý", "mô tả": "Resize ảnh về 256×256, chuẩn hóa theo ImageNet", "shape": str(tuple(result["image"].size))},
        {"bước": 2, "khối": "ResNet101 headless", "mô tả": "Trích xuất bản đồ đặc trưng toàn cục", "shape": str(trace["global_feature_map_shape"])},
        {"bước": 3, "khối": "Adaptive pooling", "mô tả": "Đưa feature map về grid cố định, rồi flatten thành global slots", "shape": f"{trace['global_slots']} × 2048"},
        {"bước": 4, "khối": "YOLOv12", "mô tả": "Phát hiện vùng vật thể, lấy tối đa TOP_K object boxes", "shape": f"{trace['valid_objects'][0]}/{trace['object_slots']} object thật"},
        {"bước": 5, "khối": "ROI Align + Projection + SFS5D", "mô tả": "Biến mỗi vùng vật thể thành vector 2048D và cộng mã hóa hình học", "shape": f"{trace['object_slots']} × 2048"},
        {"bước": 6, "khối": "Concat + Padding mask", "mô tả": "Ghép global slots và object slots, che object slot rỗng", "shape": str(trace["encoder_out_shape"])},
        {"bước": 7, "khối": "Adaptive Attention LSTM", "mô tả": "Mỗi từ sinh ra một vector context mới bằng tổng có trọng số", "shape": f"{len(steps)} bước"},
    ]
    st.dataframe(pd.DataFrame(pipeline), use_container_width=True, hide_index=True)
    st.markdown("#### Metadata checkpoint")
    st.json(loaded.checkpoint_meta)


with tabs[5]:
    st.markdown("### So sánh mô hình của nhóm với baseline hiện đại")
    st.caption(
        "Bảng này phục vụ báo cáo cuối kỳ: so số tham số, bộ nhớ tham số ước lượng, thời gian suy luận và caption sinh ra trên cùng một ảnh. "
        "Để có kết luận chính thức, hãy chạy thêm benchmark_sota_folder.py trên một subset TEST cố định."
    )

    ours_caption = beam["caption"] if beam is not None else greedy["caption"]
    ours_row = ours_comparison_row(loaded, ours_caption, result["timing"]["total_ms"])
    comparison_rows = [ours_row]

    st.markdown("#### Mô hình của nhóm")
    st.dataframe(pd.DataFrame(rows_for_display(comparison_rows)), use_container_width=True, hide_index=True)

    if not enable_sota:
        st.info("Bật checkbox 'Bật tab so sánh BLIP/GIT/BLIP-2' ở sidebar để tải và chạy các mô hình so sánh.")
    else:
        if any(SOTA_MODELS[k].heavy for k in sota_selected):
            st.warning("Bạn đã chọn baseline nặng như BLIP-2. Nếu máy không đủ GPU/RAM, hãy bỏ chọn BLIP-2 và chỉ chạy BLIP-base/GIT-base.")

        refs_for_sota = st.text_area(
            "Caption tham chiếu cho bảng so sánh một ảnh (không bắt buộc)",
            height=110,
            placeholder="a dog is running through the grass\na brown dog runs in a field",
            key="sota_refs",
        )

        if st.button("Chạy các mô hình so sánh trên ảnh này", key="run_sota_models"):
            all_rows = [ours_row]
            for key in sota_selected:
                spec = SOTA_MODELS[key]
                with st.spinner(f"Đang tải/chạy {spec.display_name}..."):
                    try:
                        baseline = cached_sota_model(key, device, sota_dtype)
                        row = generate_sota_caption(
                            baseline,
                            pil_img,
                            num_beams=beam_size,
                            max_new_tokens=sota_tokens,
                        )
                        all_rows.append(row)
                    except Exception as e:
                        st.error(f"Không chạy được {spec.display_name}. Có thể thiếu transformers, thiếu mạng tải model, hoặc máy không đủ RAM/GPU.")
                        st.exception(e)

            display_df = pd.DataFrame(rows_for_display(all_rows))
            st.markdown("#### Bảng so sánh nhanh")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            refs = [r.strip() for r in refs_for_sota.splitlines() if r.strip()]
            if refs:
                metric_rows = []
                for r in all_rows:
                    scores = score_single_image(r.get("caption", ""), refs)
                    metric_rows.append({
                        "Model": r.get("model", ""),
                        "BLEU-1": scores.get("bleu1"),
                        "BLEU-2": scores.get("bleu2"),
                        "BLEU-3": scores.get("bleu3"),
                        "BLEU-4": scores.get("bleu4"),
                        "METEOR": scores.get("meteor"),
                        "ROUGE-L": scores.get("rouge_l"),
                        "CIDEr": scores.get("cider"),
                    })
                st.markdown("#### Metrics minh họa cho một ảnh")
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

            export_df = pd.DataFrame(all_rows)
            st.download_button(
                "Tải CSV so sánh ảnh hiện tại",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="sota_comparison_current_image.csv",
                mime="text/csv",
            )

    st.markdown("#### Cách đưa vào báo cáo")
    st.markdown(
        """
- Bảng trong demo dùng để minh họa trên một ảnh.  
- Bảng trong báo cáo nên chạy bằng `benchmark_sota_folder.py` trên cùng một thư mục ảnh/subset TEST.  
- Khi kết luận “nhỏ hơn” hoặc “chậm/nhanh hơn”, phải ghi rõ điều kiện: thiết bị, số ảnh, beam size, max length, batch size.
"""
    )
