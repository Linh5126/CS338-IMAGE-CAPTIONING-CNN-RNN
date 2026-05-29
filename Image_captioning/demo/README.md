# Image Captioning Demo System

Demo này được thiết kế cho mô hình đồ án: **DualStreamEncoder + DecoderAdaptive**.

## 1. Chức năng

- Kéo thả ảnh và sinh caption.
- So sánh Greedy Search và Beam Search.
- Hiển thị attention heatmap theo từng từ trong caption.
- Hiển thị các vùng vật thể YOLO cung cấp cho mô hình captioning.
- Tách tỷ lệ chú ý thành 3 phần: ResNet global slots, YOLO object slots, và visual sentinel.
- Hiển thị bảng từng bước giải mã: từ sinh ra, xác suất, attention, context norm, gate mean.
- Nhập caption tham chiếu để tính BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr cho ảnh demo.
- Có nhật ký pipeline để thuyết trình: preprocess -> ResNet -> YOLO -> ROI Align -> SFS5D -> mask -> decoder.

## 2. Cấu trúc thư mục

```text
image_captioning_demo_system/
├── app.py                  # Giao diện Streamlit
├── demo_engine.py           # Load model, encode trace, greedy/beam decode
├── visualization.py         # Heatmap, object boxes, attention breakdown
├── demo_metrics.py          # Metrics cho một ảnh demo
├── models.py                # Copy từ đồ án, có chỉnh nhẹ YOLO_WEIGHTS env
├── utils.py                 # Copy từ đồ án
├── datasets.py              # Copy từ đồ án
├── requirements.txt
├── run_demo.sh
├── run_demo.bat
└── checkpoints/
    ├── BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar
    ├── WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json
    └── yolo12s.pt
```

## 3. Chuẩn bị file cần thiết

Bạn cần đặt các file sau vào thư mục `checkpoints/`:

1. Checkpoint tốt nhất của mô hình:

```text
BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar
```

2. Word map:

```text
WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json
```

3. YOLOv12 weights, ví dụ:

```text
yolo12s.pt
```

Nếu checkpoint lưu `yolo_version='yolo12s.pt'`, demo sẽ tìm file này. Nếu bạn đặt tên khác, sửa đường dẫn trong thanh bên của giao diện.

## 4. Cài đặt

Khuyến nghị dùng môi trường ảo Python 3.10+.

```bash
pip install -r requirements.txt
```

Nếu chạy trên Kaggle/Colab, bạn có thể cần cài thêm:

```bash
pip install streamlit ultralytics rouge-score pycocoevalcap
```

## 5. Chạy demo

Linux/macOS:

```bash
./run_demo.sh
```

Windows:

```bat
run_demo.bat
```

Hoặc chạy trực tiếp:

```bash
streamlit run app.py --server.fileWatcherType none
```

## 6. Cách thuyết trình demo

Khi chạy demo, bạn nên nói theo thứ tự:

1. Ảnh được resize và chuẩn hóa như lúc train.
2. ResNet101 tạo các ô đặc trưng toàn cục.
3. YOLOv12 phát hiện các vùng vật thể.
4. ROI Align lấy vector vùng, SFS5D cộng thông tin vị trí và kích thước.
5. Global slots và object slots được ghép lại; object slot rỗng bị padding mask che.
6. Decoder sinh từng từ. Tại mỗi bước, attention tạo một vector ngữ cảnh mới.
7. Heatmap cho thấy từ hiện tại đang dựa vào vùng ảnh nào.
8. Nếu nhập caption tham chiếu, demo tính metrics để minh họa cách đánh giá.

## 7. Lưu ý về metrics

Metrics trong giao diện chỉ dùng cho **một ảnh** để minh họa. Báo cáo chính thức nên dùng `eval.py` trên toàn bộ tập TEST, vì BLEU, METEOR, ROUGE-L và CIDEr có ý nghĩa ổn định hơn ở mức corpus.

## 8. Lưu ý về heatmap

Heatmap được bám theo đường Greedy Search vì Greedy là một chuỗi quyết định duy nhất, dễ giải thích từng từ. Beam Search giữ nhiều giả thuyết cùng lúc nên việc gán một heatmap duy nhất cho toàn bộ quá trình beam sẽ dễ gây hiểu nhầm.

## 9. Khi gặp lỗi YOLO

Nếu lỗi liên quan `yolo12s.pt`, hãy kiểm tra:

- File `yolo12s.pt` có nằm trong `checkpoints/` không.
- Đường dẫn trong ô “YOLO weights” trên sidebar có đúng không.
- Máy có internet không nếu muốn Ultralytics tự tải weights.

Trong demo này, tốt nhất là đặt sẵn file `.pt` để thuyết trình không phụ thuộc mạng.

---

## Bổ sung v4: So sánh với BLIP/GIT/BLIP-2

Bản v4 có thêm:

- `sota_compare.py`: load/generate/count parameters cho BLIP-base, GIT-base-COCO, BLIP-2 OPT-2.7B.
- Tab **So sánh SOTA** trong `app.py`.
- `benchmark_sota_folder.py`: chạy benchmark trên nhiều ảnh và xuất CSV cho báo cáo.
- `README_SOTA_BENCHMARK.md`: hướng dẫn chi tiết.

Khuyến nghị chạy baseline theo thứ tự:

1. BLIP-base
2. GIT-base-COCO
3. BLIP-2 OPT-2.7B nếu máy đủ mạnh

Mục tiêu của bảng so sánh là chứng minh bằng số liệu: số tham số, bộ nhớ tham số, thời gian suy luận và chênh lệch chất lượng caption, thay vì chỉ nói miệng “gọn nhẹ”.
