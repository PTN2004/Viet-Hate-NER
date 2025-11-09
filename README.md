# 🧠 Hate Speech Detection API (Vietnamese NER)

Một API nhận diện **ngôn từ thù ghét và xúc phạm** trong tiếng Việt (Hate & Offensive Speech Detection),
được huấn luyện dựa trên mô hình **PhoBERT** và tập dữ liệu **ViHOS (Vietnamese Hate and Offensive Spans Detection)**.

Triển khai với **FastAPI**, tương thích **Docker**, và sẵn sàng **deploy** lên bất kỳ hạ tầng nào.

---

## 🚀 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Chạy API](#-chạy-api)
- [Gọi thử API](#-gọi-thử-api)
- [Ví dụ kết quả](#-ví-dụ-kết-quả)
- [Docker](#-docker)
- [Hướng phát triển](#-hướng-phát-triển)
- [Giấy phép](#-giấy-phép)

---

## 🧩 Giới thiệu

Dự án này cung cấp một **REST API** cho bài toán **Named Entity Recognition (NER)**
nhằm phát hiện **từ, cụm từ thù ghét hoặc xúc phạm** trong văn bản tiếng Việt.

Model được huấn luyện trên **PhoBERT-base** với nhãn dữ liệu dạng **BIO (Begin-Inside-Outside)**.

**Ví dụ nhãn:**
| Token | Label |
|--------|--------|
| Con | O |
| nhỏ | B-HATE |
| ngu | B-HATE |
| thật | I-HATE |
| quá | O |


---

## ⚙️ Cài đặt

### 1️⃣ Clone dự án
```bash
git clone https://github.com/PTN2004/Viet-Hate-NER.git
cd viet-hate-ner
```


### 2️⃣ Cài thư viện

Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\\Scripts\\activate     # Windows
```

Cài dependencies:

```bash
pip install -r requirements.txt
```
---
## 🧠 Chạy API

Nếu bạn dùng cấu trúc 1 file (main.py):
```bash
python app/main.py
```

hoặc:

```bash
uvicorn app.main:create_app --factory --reload --port 8000
```

Mở trình duyệt tại 👉 http://localhost:8000/docs

---
## 🧠 Hướng phát triển

| Tính năng           | Mô tả                                               |
| ------------------- | --------------------------------------------------- |
| 🧩 Batch prediction | Hỗ trợ xử lý nhiều câu cùng lúc                     |
| 🧠 Multi-model      | Cho phép chọn mô hình (hate / sentiment / toxicity) |
| 📈 Logging          | Theo dõi request và hiệu năng inference             |
| 🐍 CLI              | Gọi mô hình qua dòng lệnh                           |
| 🌐 Deploy           | Đưa API lên Render / HuggingFace Spaces / VPS       |
