# So sánh mô hình của nhóm với BLIP/GIT/BLIP-2

Bản demo v4 bổ sung tab **So sánh SOTA** và script `benchmark_sota_folder.py` để lấy số liệu đưa vào báo cáo cuối cùng.

## 1. Cài đặt

Khuyến nghị dùng Python 3.10 hoặc 3.11. Nếu dùng Python 3.14 mà lỗi PyTorch/Transformers, hãy tạo môi trường mới.

```bash
pip install -r requirements.txt
```

Lần đầu chạy BLIP/GIT/BLIP-2 cần internet để tải model từ Hugging Face. Sau khi tải xong, model sẽ được cache trên máy.

## 2. Chạy trong giao diện Streamlit

```bash
streamlit run app.py --server.fileWatcherType none
```

Trong sidebar:

1. Bật **Bật tab so sánh BLIP/GIT/BLIP-2**.
2. Chọn `BLIP-base` và `GIT-base-COCO` trước.
3. Chỉ chọn `BLIP-2 OPT-2.7B` nếu máy có đủ GPU/RAM.
4. Tải một ảnh lên, sau đó mở tab **So sánh SOTA**.
5. Nhấn **Chạy các mô hình so sánh trên ảnh này**.

Tab này sẽ hiển thị:

- caption của mô hình nhóm và các baseline,
- tổng số tham số,
- số tham số đang trainable theo trạng thái model đang load,
- bộ nhớ tham số ước lượng,
- thời gian suy luận trên ảnh hiện tại,
- VRAM đỉnh nếu dùng CUDA.

## 3. Chạy benchmark trên nhiều ảnh để đưa vào báo cáo

Đặt một số ảnh test vào thư mục, ví dụ `sample_inputs/`.

```bash
python benchmark_sota_folder.py \
  --checkpoint checkpoints/checkpoint.pth.tar \
  --word-map checkpoints/word_map.json \
  --yolo checkpoints/yolo12s.pt \
  --image-dir sample_inputs \
  --models blip_base git_base_coco \
  --output benchmark_results.csv
```

Nếu máy đủ mạnh, có thể thêm BLIP-2:

```bash
python benchmark_sota_folder.py \
  --checkpoint checkpoints/checkpoint.pth.tar \
  --word-map checkpoints/word_map.json \
  --yolo checkpoints/yolo12s.pt \
  --image-dir sample_inputs \
  --models blip_base git_base_coco blip2_opt_2_7b \
  --output benchmark_results.csv
```

## 4. Cách viết trong báo cáo

Không nên viết chung chung “mô hình nhóm gọn nhẹ hơn”. Hãy viết theo dạng có số liệu:

> Trên cùng thiết bị, cùng beam size và cùng số ảnh test, mô hình nhóm có X tham số, dùng Y MB bộ nhớ tham số, suy luận trung bình Z ms/ảnh. BLIP-base có ..., GIT-base-COCO có ... Vì vậy mô hình nhóm thấp hơn/cao hơn về chất lượng caption nhưng có/không có lợi thế về quy mô và chi phí suy luận.

Cần ghi rõ:

- thiết bị chạy,
- số ảnh benchmark,
- batch size,
- beam size,
- max new tokens / max length,
- model nào được chạy trực tiếp, model nào chỉ dùng số liệu tham khảo.

## 5. Lưu ý

- BLIP-base và GIT-base-COCO là baseline hiện đại dễ chạy trên máy cá nhân hơn BLIP-2/LLaVA.
- BLIP-2 OPT-2.7B rất nặng; nếu máy không đủ, hãy dùng nó như mốc tham khảo quy mô từ tài liệu/model card, hoặc chạy trên Colab/Kaggle GPU mạnh.
- Bảng trong tab demo chỉ là so sánh một ảnh. Báo cáo nên dùng kết quả CSV từ `benchmark_sota_folder.py` trên một subset cố định.
