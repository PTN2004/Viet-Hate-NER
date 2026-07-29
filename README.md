# 🧠 Vietnamese Hate Speech Detection API (NER)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Một hệ thống API hiệu năng cao chuyên nhận diện **ngôn từ thù ghét và xúc phạm** trong tiếng Việt (Hate & Offensive Speech Detection). 
Mô hình lõi sử dụng kiến trúc **PhoBERT-base**, được tinh chỉnh (fine-tuned) cho bài toán Trích xuất thực thể có tên (Named Entity Recognition - NER) trên tập dữ liệu **ViHOS** (Vietnamese Hate and Offensive Spans Detection).

---

## 🏆 Điểm nổi bật (Key Achievements)

*   **Vượt qua Baseline chính thức:** Mô hình đạt **F1-score 78.81%**, vượt qua hoàn toàn các baseline SOTA được công bố trong bài báo gốc của ViHOS (bao gồm cả XLM-R Large với 77.70% và PhoBERT Base gốc với 75.69%).
*   **Hiệu năng Suy luận (Inference Speed):** API được tối ưu hóa để đạt tốc độ xử lý xấp xỉ **~97 samples/second**, đảm bảo độ trễ thấp (low-latency) cho các ứng dụng kiểm duyệt nội dung thời gian thực.
*   **MLOps & Deployment:** Đóng gói hoàn chỉnh bằng Docker, tích hợp Swagger UI tự động.

## 📊 Hiệu suất Mô hình (Model Performance)

Đánh giá trên tập Evaluation của ViHOS Dataset:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 82.36% |
| **F1-Score** | **78.81%** |
| **Precision** | 78.39% |
| **Recall** | 82.35% |
| **Inference Speed** | ~96.84 samples/sec |

*So sánh với Baseline bài báo ViHOS:*
*   *XLM-R (Large): 77.70%*
*   *PhoBERT (Base): 75.69%*
*   ***This Repo (Fine-tuned PhoBERT-base): 78.81%***

---

## ⚙️ Hướng dẫn Cài đặt & Triển khai

### 1️⃣ Chạy bằng Docker (Khuyến nghị cho Production)
Cách nhanh nhất để khởi chạy API mà không cần lo lắng về môi trường:

```bash
# Clone dự án
git clone [https://github.com/PTN2004/Viet-Hate-NER.git](https://github.com/PTN2004/Viet-Hate-NER.git)
cd viet-hate-ner

# Build image và chạy container
docker build -t viet-hate-ner-api .
docker run -d -p 8000:8000 viet-hate-ner-api

```

### 2️⃣ Chạy Local (Cho quá trình Development)

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt

# Khởi chạy server FastAPI
uvicorn app.main:create_app --factory --reload --port 8000

```

---

## 🚀 Hướng dẫn Sử dụng API

Sau khi khởi chạy, truy cập tài liệu API tương tác tại: 👉 **http://localhost:8000/docs**

### Ví dụ minh họa (BIO Tagging)

Hệ thống sử dụng nhãn dạng **BIO (Begin-Inside-Outside)** để trích xuất chính xác vị trí từ vựng độc hại:

| Token | Label | Ý nghĩa |
| --- | --- | --- |
| Mấy | O | Ngoài cụm độc hại |
| thằng | B-HATE | Bắt đầu cụm độc hại |
| khờ | I-HATE | Bên trong cụm độc hại |
| này | O | Ngoài cụm độc hại |

---

## 📈 Lộ trình Phát triển (Future Work)

| Tính năng | Trạng thái | Mô tả |
| --- | --- | --- |
| 🧩 **Batch Prediction** | ⏳ Planned | Hỗ trợ xử lý pipeline nhiều câu cùng lúc để tăng thông lượng (throughput). |
| 📉 **Model Quantization** | ⏳ Planned | Áp dụng FP16/INT8 để giảm dung lượng mô hình và tăng tốc CPU inference. |
| 📈 **Monitoring & Logging** | ⏳ Planned | Tích hợp Prometheus/Grafana để theo dõi API metrics. |
| 🌐 **Cloud Deployment** | ⏳ Planned | Đưa API lên Render / HuggingFace Spaces. |

---

*Developed by Pham Ngoc Tu*
