# 🤖 Ứng dụng Chatbot Hỗ trợ Học sinh THCS Học Toán  

## 👋 Giới thiệu  
Trong bối cảnh giáo dục 4.0, việc tích hợp **trí tuệ nhân tạo (AI)** vào quá trình dạy và học ngày càng quan trọng.  
Dự án này xây dựng một **chatbot thông minh bằng tiếng Việt** giúp học sinh **trung học cơ sở** học môn **Toán** hiệu quả hơn, với các chức năng:  
- Trả lời câu hỏi lý thuyết.  
- Gợi ý và chấm bài tập theo lớp, chủ đề, thể loại.  
- Phản hồi kết quả và đưa ra hướng dẫn học tập.  

Ứng dụng sử dụng các công nghệ NLP hiện đại như **PhoBERT, FAISS, Fuzzy Matching**, kết hợp với giao diện **Streamlit** thân thiện, dễ dùng.  

---

## 🎯 Mục tiêu  
- Xây dựng bộ dữ liệu Toán học (lớp 6–9) gồm lý thuyết, bài tập, đáp án, hướng dẫn giải.  
- Tích hợp mô hình **PhoBERT** để hiểu ngữ nghĩa câu hỏi tiếng Việt.  
- Triển khai tìm kiếm thông minh bằng **FAISS** (tìm kiếm vector) và **Fuzzy Matching** (so khớp ký tự).  
- Thiết kế giao diện đơn giản, thân thiện cho học sinh.  

---

## 📚 Bộ dữ liệu  
- Hơn **1100 câu hỏi** và bài tập Toán (lớp 6–9).  
- Phân loại theo: **Lớp, Chủ đề, Thể loại, Câu hỏi, Đáp án, Hướng dẫn giải**.  
- Nguồn: Sách giáo khoa + các trang giáo dục uy tín (hocmai.vn, vietjack.com, loigiaihay.com, kenhgiaovien.com, …).  

---

## ⚙️ Công nghệ sử dụng  
- **Ngôn ngữ**: Python  
- **Xử lý ngôn ngữ tự nhiên**: [PhoBERT](https://github.com/VinAIResearch/PhoBERT)  
- **Tìm kiếm vector**: [FAISS](https://github.com/facebookresearch/faiss)  
- **So khớp mờ**: Fuzzy Matching (fuzzywuzzy)  
- **Giao diện**: [Streamlit](https://streamlit.io)  
- **Thư viện khác**: pandas, numpy, scikit-learn, matplotlib  

---

## 🏗️ Kiến trúc hệ thống  
1. **Giao diện người dùng (Streamlit)**  
   - Cho phép nhập câu hỏi.  
   - Hiển thị câu trả lời, bài tập, và phản hồi.  

2. **Bộ xử lý truy vấn**  
   - Phân tích câu hỏi → PhoBERT sinh embedding.  
   - Tìm kiếm câu hỏi tương đồng bằng FAISS.  
   - Nếu không đủ độ chính xác → dùng Fuzzy Matching.  

3. **Hệ thống dữ liệu**  
   - Bộ dữ liệu Toán lớp 6–9 được lưu trữ dưới dạng CSDL.  

---

## 🧪 Kết quả thực nghiệm  
- **Top-1 Accuracy**: 99.74% (1168/1171) → Chatbot gần như luôn đưa đúng câu trả lời ở vị trí đầu tiên.  
- **Recall@3**: 100% → Đảm bảo câu trả lời đúng nằm trong top 3.  
- **Precision@3**: 33.33% → Gợi ý top 3 vẫn còn nhiễu.  

---

## 💻 Demo giao diện  
👉 Trải nghiệm trực tiếp tại: [Streamlit App](https://hecwdkaavo9t2pdu49nuan.streamlit.app/)  

### Giao diện hỏi lý thuyết  
- Học sinh nhập câu hỏi.  
- Chatbot tìm kiếm và trả lời.  

### Giao diện yêu cầu bài tập  
- Học sinh chọn: **Lớp → Chủ đề → Thể loại**.  
- Chatbot đưa ra bài tập, kiểm tra câu trả lời và phản hồi.  

---

## 📌 Cách chạy dự án  

```bash
# Clone repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
