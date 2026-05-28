# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 đến 1.5, các phản hồi trở nên ngày càng đa dạng, sáng tạo nhưng ít nhất quán hơn. Temperature 0.0 cho phản hồi xác định và lặp lại, trong khi temperature cao hơn (1.0-1.5) tạo ra các sự thật hoặc diễn giải khác nhau. Điều này phản ánh rằng temperature kiểm soát độ ngẫu nhiên trong quá trình sinh phát.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature 0.3-0.5 cho chatbot hỗ trợ khách hàng vì nó cần phải nhất quán, đáng tin cậy, và cung cấp thông tin chính xác để giải quyết vấn đề của người dùng. Quá cao temperature sẽ khiến các phản hồi thay đổi hoặc không chính xác, làm mất lòng tin của khách hàng.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Với 10,000 người × 3 lần/người × 350 token = 10.5 triệu token/ngày. Giả sử chia đều input/output (~175 token mỗi cái):
> - GPT-4o: (175 × $5 + 175 × $20) / 1M × 10.5M = $4,412.50/ngày
> - GPT-4o-mini: (175 × $0.15 + 175 × $0.6) / 1M × 10.5M = $13.13/ngày
> GPT-4o đắt hơn khoảng **336 lần** cho workload này.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> **GPT-4o xứng đáng**: Ứng dụng tài chính, pháp lý, hay y tế nơi mà độ chính xác cao hơn 10% có thể tiết kiệm hàng triệu đô, hoặc khi xử lý các tác vụ phức tạp như phân tích chính sách hoặc tư vấn chiến lược. 
> **GPT-4o-mini tốt hơn**: Chatbot hỗ trợ khách hàng, dịch thuật đơn giản, tóm tắt văn bản, hoặc các ứng dụng quy mô lớn với yêu cầu chi phí thấp nhưng chất lượng chấp nhận được.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming là cần thiết khi người dùng mong đợi phản hồi tức thì hoặc dài dòng, chẳng hạn như chatbot thời gian thực, trợ lý viết hỗ trợ (code generation), hoặc các ứng dụng web interactiveở nơi hiển thị từng token tạo cảm giác ứng dụng đang "suy nghĩ". Non-streaming phù hợp hơn khi: (1) phản hồi ngắn và có thể đợi, (2) cần xử lý toàn bộ phản hồi trước khi hiển thị (ví dụ: structured data extraction), (3) kết nối ненадежна hoặc client không hỗ trợ streaming, hoặc (4) chi phí overhead kết nối streaming cao hơn cải thiện trải nghiệm.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
