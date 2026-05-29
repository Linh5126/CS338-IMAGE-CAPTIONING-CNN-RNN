# CS338.Q22 – Image Captioning using CNN–RNN/LSTM

> Đồ án môn **Nhận dạng – CS338.Q22**  
> Chủ đề: **Sinh mô tả ảnh tự động (Image Captioning)**  
> Nhóm 17: **Nguyễn An Trần Đồng – 23520298** và **Lê Xuân Song Lĩnh – 23520845**  
> Giảng viên hướng dẫn: **TS. Dương Việt Hằng**

---

## 1. Giới thiệu

Dự án này xây dựng và đánh giá một hệ thống **image captioning** có khả năng sinh câu mô tả tự nhiên từ ảnh đầu vào. Điểm xuất phát của đồ án là mô hình CNN–RNN/LSTM có attention trong repository [`sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning). Từ nền tảng đó, nhóm phát triển thêm các thành phần nhằm tăng khả năng hiểu vật thể, giữ thông tin không gian và cải thiện chất lượng caption ở cấp toàn câu.

Trọng tâm của đồ án không phải là dùng một mô hình thị giác-ngôn ngữ rất lớn như một hộp đen, mà là xây dựng một pipeline có thể phân tích được: ảnh được mã hóa như thế nào, đặc trưng vật thể được đưa vào ra sao, attention chọn vùng nào ở từng bước sinh từ, và quá trình tối ưu theo metric cấp câu ảnh hưởng như thế nào đến kết quả.

---

## 2. Mục tiêu nghiên cứu

Dự án tập trung vào các câu hỏi sau:

1. **Dual Encoder có giúp mô hình kết hợp tốt hơn bối cảnh toàn ảnh và vật thể cụ thể không?**  
   ResNet101 cung cấp đặc trưng toàn cục, trong khi YOLOv12 cung cấp các vùng vật thể nổi bật.

2. **Thông tin vị trí của bounding box có giúp object feature giàu thông tin hơn không?**  
   Spatial Fusion 5D mã hóa tâm, kích thước và diện tích tương đối của bounding box để đưa vào object feature.

3. **Decoder có cần luôn nhìn vào ảnh ở mọi bước sinh từ không?**  
   Adaptive Attention với Visual Sentinel cho phép decoder chọn giữa vùng ảnh và ngữ cảnh ngôn ngữ bên trong LSTM.

4. **Tối ưu theo từng token có đủ cho image captioning không?**  
   SCST được dùng để fine-tune mô hình theo CIDEr reward, giảm khoảng cách giữa huấn luyện bằng cross-entropy và đánh giá bằng metric cấp câu.

5. **Mô hình đề xuất còn cách các mô hình pretrained hiện đại bao xa?**  
   Nhóm chạy BLIP-base, GIT-base-COCO và BLIP-2 OPT-2.7B trên cùng tập Flickr8k TEST để so sánh chất lượng và quy mô tham số.

---

## 3. Đóng góp chính

Dự án có các đóng góp chính sau:

- Xây dựng mô hình **Dual-Stream Encoder** kết hợp:
  - **ResNet101 global stream** cho bối cảnh toàn ảnh.
  - **YOLOv12 object stream** cho vùng vật thể cụ thể.

- Tích hợp **Spatial Fusion 5D** để đưa thông tin hình học của bounding box vào object feature.

- Tích hợp **Adaptive Attention with Visual Sentinel** để decoder có thể quyết định khi nào cần nhìn ảnh và khi nào nên dựa vào ngữ cảnh ngôn ngữ.

- Fine-tune mô hình bằng **Self-Critical Sequence Training (SCST)** với CIDEr reward.

- Xây dựng demo Streamlit có khả năng:
  - sinh caption bằng Greedy Search và Beam Search,
  - hiển thị YOLO boxes,
  - hiển thị attention heatmap theo từng từ,
  - phân tách attention thành global slots, object slots và visual sentinel,
  - so sánh caption với các mô hình BLIP/GIT/BLIP-2.

- Bổ sung benchmark với các mô hình pretrained hiện đại trên cùng tập Flickr8k TEST.

---

## 4. Kết quả chính

### 4.1. Kết quả mô hình cuối trên Flickr8k TEST

Mô hình cuối sử dụng **Dual Encoder + Adaptive Attention + Spatial Fusion 5D + SCST** và giải mã bằng **Beam Search với beam size = 5**.

| Model | Decode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ours Final | Beam5 | 0.7239 | 0.5503 | 0.4092 | 0.2934 | 0.4716 | 0.5241 | 0.7702 |

### 4.2. So sánh với mô hình pretrained hiện đại

Các mô hình BLIP-base, GIT-base-COCO và BLIP-2 OPT-2.7B được dùng ở chế độ **pretrained inference**, không fine-tune lại trên Flickr8k. Bảng dưới đây phản ánh mức chênh lệch khi chạy suy luận trên cùng tập TEST, không phải so sánh công bằng về điều kiện huấn luyện từ đầu.

| Model | Params | So với mô hình đề xuất | BLEU-4 | CIDEr |
|---|---:|---:|---:|---:|
| Ours Final | 67.35M | 1.00× | 0.2934 | 0.7702 |
| BLIP-base | 247.44M | 3.67× | 0.2852 | 0.8446 |
| GIT-base-COCO | 176.62M | 2.62× | 0.2093 | 0.5019 |
| BLIP-2 OPT-2.7B | 3744.76M | 55.60× | 0.3355 | 0.9970 |

Mô hình của nhóm nhỏ hơn BLIP-2 OPT-2.7B khoảng **55.60 lần**, đạt khoảng **87.45% BLEU-4** và **77.25% CIDEr** so với BLIP-2 trên Flickr8k TEST. Kết quả cho thấy mô hình chưa đạt chất lượng của các mô hình ảnh-ngôn ngữ lớn, nhưng vẫn giữ được khả năng phân tích từng module như global slots, object slots, padding mask, attention heatmap và visual sentinel.

---

## 5. Cấu trúc repository

```text
CS338_Q22_GROUP_17/
├── Report_CS338.pdf                         # Báo cáo cuối
├── Report_CS338.docx                        # Bản Word của báo cáo
├── CS338.pdf                                # Slide thuyết trình
├── demo.zip                                 # Demo Streamlit và benchmark SOTA
├── code/
│   ├── benchmark_sota_flickr8k_test.ipynb   # Notebook benchmark BLIP/GIT/BLIP-2 trên TEST
│   ├── create_input_files.py                # Tạo HDF5/JSON từ Flickr8k
│   ├── caption.py                           # Captioning script theo hướng baseline
│   ├── val.ipynb                            # Notebook kiểm thử/đánh giá
│   ├── train_02.ipynb                       # Notebook huấn luyện
│   ├── dataset_flickr8k.json                # Metadata/split captions
│   ├── WORDMAP_*.json                       # Word map
│   ├── TRAIN_CAPTIONS_*.json
│   ├── VAL_CAPTIONS_*.json
│   └── TEST_CAPTIONS_*.json
├── model/
│   ├── resnet/                              # Baseline ResNet-LSTM
│   ├── yolo/                                # Dual Encoder với YOLO stream
│   ├── yolo+ap/                             # Dual Encoder + Adaptive Attention
│   ├── yolo+ap+f/                           # Dual Encoder + Adaptive + Spatial Fusion 5D
│   └── rl/                                  # Final model + SCST
└── README.md
```

Trong mỗi thư mục con của `model/` thường có:

```text
models.py       # Định nghĩa encoder/decoder
train.py        # Huấn luyện
_eval.py/eval.py # Đánh giá
utils.py        # Hàm tiện ích, checkpoint, preprocessing
datasets.py     # Dataset loader
```

---

## 6. Môi trường cài đặt

Khuyến nghị:

- Python 3.10 hoặc 3.11
- PyTorch 2.x
- CUDA GPU nếu huấn luyện hoặc chạy BLIP-2
- Google Colab/Kaggle GPU được khuyến nghị cho benchmark SOTA

Cài đặt các thư viện chính:

```bash
pip install torch torchvision torchaudio
pip install ultralytics nltk rouge-score pycocoevalcap h5py tqdm pandas matplotlib pillow
pip install transformers accelerate safetensors sentencepiece
```

Nếu chạy demo Streamlit, giải nén `demo.zip` rồi cài theo `requirements.txt` trong thư mục demo:

```bash
unzip demo.zip -d demo
cd demo
pip install -r requirements.txt
```

---

## 7. Chuẩn bị dữ liệu

Dự án sử dụng **Flickr8k**. Trong cấu hình thực nghiệm, split thường dùng gồm:

- 6000 ảnh train
- 1000 ảnh validation
- 1000 ảnh test
- 5 caption tham chiếu cho mỗi ảnh

Dữ liệu sau tiền xử lý gồm các file dạng:

```text
TRAIN_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
VAL_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5
TRAIN_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
VAL_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
TEST_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json
WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json
```

Tạo input files từ Flickr8k:

```bash
cd code
python create_input_files.py --dataset flickr8k
```

Lưu ý: cần chỉnh đường dẫn `json_path`, `img_folder` và `output_folder` trong `create_input_files.py` cho đúng máy của bạn.

---

## 8. Huấn luyện các biến thể mô hình

Các biến thể chính nằm trong thư mục `model/`.

### 8.1. Baseline ResNet-LSTM

```bash
cd model/resnet
python train.py
```

### 8.2. Dual Encoder với YOLOv12

```bash
cd model/yolo
python train.py
```

### 8.3. Dual Encoder + Adaptive Attention

```bash
cd model/yolo+ap
python train.py
```

### 8.4. Dual Encoder + Adaptive Attention + Spatial Fusion 5D

```bash
cd model/yolo+ap+f
python train.py
```

### 8.5. Fine-tuning bằng SCST

```bash
cd model/rl
python train.py
```

Trước khi chạy, cần kiểm tra trong `train.py`:

```python
data_folder = '...'
checkpoint = '...'
```

Các đường dẫn này cần trỏ đúng tới thư mục chứa dữ liệu HDF5/JSON và checkpoint tương ứng.

---

## 9. Đánh giá mô hình

Script đánh giá hỗ trợ Greedy Search, Beam Search và các metric BLEU, METEOR, ROUGE-L, CIDEr.

Ví dụ đánh giá mô hình cuối:

```bash
cd model/rl
python eval.py \
  --checkpoint /path/to/BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar \
  --data-folder /path/to/data_folder \
  --data-name flickr8k_5_cap_per_img_5_min_word_freq \
  --split TEST \
  --decode both \
  --beam-size 5 \
  --length-penalty-alpha 0.7 \
  --save-json final_predictions.json
```

Trong đánh giá, mỗi ảnh TEST chỉ được sinh một hypothesis, sau đó hypothesis này được so với toàn bộ 5 caption tham chiếu của ảnh đó. Điều này tránh lỗi đánh giá lặp cùng một ảnh 5 lần.

---

## 10. Chạy demo Streamlit

Giải nén `demo.zip`:

```bash
unzip demo.zip -d demo
cd demo
pip install -r requirements.txt
```

Chuẩn bị thư mục `checkpoints/`:

```text
demo/checkpoints/
├── BEST_checkpoint_dual_stream_lstm_adaptive_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar
├── WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json
└── yolo12s.pt
```

Chạy demo:

```bash
streamlit run app.py --server.fileWatcherType none
```

Trên Windows có thể chạy:

```bat
run_demo.bat
```

Demo hỗ trợ:

- upload ảnh bất kỳ,
- sinh caption bằng Greedy và Beam Search,
- hiển thị attention heatmap theo từng từ,
- hiển thị YOLO object boxes,
- phân tách attention thành global/object/sentinel,
- nhập reference caption để tính metric minh họa cho một ảnh,
- so sánh caption với BLIP-base, GIT-base-COCO và BLIP-2 nếu phần cứng cho phép.

---

## 11. Benchmark với BLIP/GIT/BLIP-2

Notebook benchmark nằm tại:

```text
code/benchmark_sota_flickr8k_test.ipynb
```

Mục tiêu của notebook là chạy các mô hình pretrained hiện đại trên cùng tập Flickr8k TEST để đo:

- BLEU-1/2/3/4,
- METEOR,
- ROUGE-L,
- CIDEr,
- số tham số,
- thời gian suy luận,
- VRAM đỉnh nếu dùng GPU.

Các mô hình được dùng:

- `Salesforce/blip-image-captioning-base`
- `microsoft/git-base-coco`
- `Salesforce/blip2-opt-2.7b`

Lưu ý: các mô hình này được dùng ở chế độ pretrained inference, không fine-tune lại trên Flickr8k. Vì vậy, kết quả là benchmark suy luận trên cùng tập TEST, không phải so sánh công bằng về điều kiện huấn luyện từ đầu.

---

## 12. Lưu ý về checkpoint

Checkpoint mô hình cuối có thể có dung lượng lớn và không nhất thiết được lưu trực tiếp trong repository GitHub. Nếu không có checkpoint, demo sẽ báo lỗi không tìm thấy file `.pth` hoặc `.pth.tar`.

Cần đặt checkpoint vào:

```text
demo/checkpoints/
```

hoặc chỉnh đường dẫn trong giao diện Streamlit/sidebar.

Nếu gặp lỗi:

```text
ModuleNotFoundError: No module named 'models'
```

hãy đảm bảo file `models.py` nằm cùng thư mục chạy notebook/script, hoặc thêm thư mục project vào `sys.path`.

Nếu gặp lỗi:

```text
ModuleNotFoundError: No module named 'ultralytics'
```

hãy cài:

```bash
pip install ultralytics
```

---

## 13. Hạn chế

Một số hạn chế chính của dự án:

- Flickr8k là dataset nhỏ, dễ làm mô hình sinh caption chung chung.
- YOLOv12 được dùng ở trạng thái frozen, nên object features không được tối ưu trực tiếp theo captioning loss.
- LSTM decoder khó cạnh tranh với Transformer decoder hoặc các mô hình thị giác-ngôn ngữ lớn.
- Metric tự động như BLEU/CIDEr không thay thế hoàn toàn đánh giá của con người.
- Benchmark với BLIP/GIT/BLIP-2 dùng checkpoint pretrained, nên không phải so sánh công bằng về điều kiện huấn luyện.

---

## 14. Hướng phát triển

Các hướng phát triển tiếp theo:

- Thử Transformer decoder hoặc multi-head cross-attention.
- Fine-tune một phần YOLO hoặc thêm object labels như anchor semantics.
- Huấn luyện trên dataset lớn hơn như Flickr30k hoặc MS COCO.
- So sánh với đặc trưng ảnh-ngôn ngữ từ CLIP/BLIP.
- Thực hiện human study để đánh giá độ đúng với ảnh, độ trôi chảy và độ cụ thể của caption.
- Phân tích lỗi có hệ thống theo nhóm lỗi: sai vật thể, sai hành động, thiếu chi tiết, hallucination.

---

## 15. Tài liệu tham khảo chính

1. O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. **Show and Tell: A Neural Image Caption Generator**. CVPR, 2015.
2. K. Xu et al. **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**. ICML, 2015.
3. K. He et al. **Deep Residual Learning for Image Recognition**. CVPR, 2016.
4. J. Lu et al. **Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning**. CVPR, 2017.
5. P. Anderson et al. **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering**. CVPR, 2018.
6. S. J. Rennie et al. **Self-Critical Sequence Training for Image Captioning**. CVPR, 2017.
7. R. Vedantam et al. **CIDEr: Consensus-based Image Description Evaluation**. CVPR, 2015.
8. J. Li et al. **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**. ICML, 2022.
9. J. Wang et al. **GIT: A Generative Image-to-text Transformer for Vision and Language**. 2022.
10. J. Li et al. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. ICML, 2023.
11. Y. Tian, Q. Ye, and D. Doermann. **YOLOv12: Attention-Centric Real-Time Object Detectors**. arXiv, 2025.
12. S. Vinod. **a-PyTorch-Tutorial-to-Image-Captioning**. GitHub repository.

---

## 16. Acknowledgement

Dự án có tham khảo cấu trúc và một phần pipeline từ repository mở `a-PyTorch-Tutorial-to-Image-Captioning` của S. Vinod. Nhóm đã mở rộng mô hình theo hướng Dual Encoder, YOLOv12 object stream, Spatial Fusion 5D, Adaptive Attention, SCST, benchmark SOTA và demo trực quan hóa.

---

## 17. Ghi chú nộp bài

Báo cáo cuối và slide thuyết trình đã được đính kèm trong repository:

```text
Report_CS338.pdf
Report_CS338.docx
CS338.pdf
```

Supplementary material bao gồm source code, demo Streamlit, notebook benchmark SOTA và các file tiền xử lý cần thiết.
