# Báo cáo mô tả dữ liệu (Dataset Description)

## 1. Tổng quan bộ dữ liệu
- **Vị trí**: Thư mục `dataset/`
- **Số lượng**: 70 tập tin văn bản (.txt), từ `doc_1.txt` đến `doc_70.txt`.
- **Tổng dung lượng**: Khoảng vài Megabytes. Dung lượng các file dao động từ vài KB đến hàng trăm KB.
- **Chủ đề chính**: Bộ dữ liệu chứa các bài báo, báo cáo và tin tức liên quan đến các công ty công nghệ và lĩnh vực xe điện (Electric Vehicles - EV), phân tích thị trường, và các thông tin kinh tế kỹ thuật khác (Tech Company Corpus).

## 2. Cấu trúc mỗi tập tin
Mỗi tập tin `.txt` đều có cấu trúc tiêu chuẩn được trích xuất từ các công cụ tìm kiếm hoặc báo cáo, bao gồm các trường:
- **Query**: Câu truy vấn tìm kiếm (VD: `US electric vehicle sector sentiment analysis`).
- **Title**: Tiêu đề của bài viết hoặc báo cáo.
- **Link**: Đường dẫn URL tới bài viết gốc.
- **Snippet**: Đoạn trích dẫn ngắn (tóm tắt) của bài viết.
- **Full Content**: Nội dung văn bản chi tiết của bài báo hoặc báo cáo. Đây là phần dữ liệu thô (raw text) quan trọng nhất sẽ được dùng để trích xuất thực thể (Entity) và quan hệ (Relation) cho hệ thống GraphRAG.

## 3. Đánh giá chất lượng và tiền xử lý (Preprocessing)
- **Đặc điểm**: Dữ liệu có dạng văn bản bán cấu trúc (chứa metadata ở đầu và nội dung thô ở dưới).
- **Tiền xử lý cần thiết cho Lab**:
  - Tách phần `Full Content` ra khỏi metadata.
  - Làm sạch các ký tự đặc biệt, dấu câu không cần thiết hoặc các liên kết web rác bên trong văn bản.
  - Chia nhỏ văn bản (Chunking) thành các đoạn có kích thước phù hợp (ví dụ: 500 - 1000 tokens) để đưa vào LLM trích xuất đồ thị tri thức nhằm tối ưu chi phí (Token usage) và tránh vượt quá Context Window.
