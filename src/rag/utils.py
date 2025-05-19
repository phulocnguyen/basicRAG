import re
def extract_answer(text_response: str,
                   pattern: str = r"Answer:\s*(.*)") -> str:
    """
    Trích xuất phần trả lời từ một chuỗi văn bản dựa trên một mẫu regular expression.

    Args:
        text_response: Chuỗi văn bản đầu vào cần trích xuất.
        pattern: Mẫu regular expression để tìm kiếm câu trả lời.
                 Mặc định là r"Answer:\s*(.*)" để tìm kiếm chuỗi bắt đầu bằng "Answer:",
                 theo sau là không hoặc nhiều khoảng trắng, và sau đó bắt lấy mọi ký tự cho đến hết dòng.

    Returns:
        Chuỗi chứa phần trả lời đã trích xuất (sau khi loại bỏ khoảng trắng ở đầu và cuối),
        hoặc chuỗi "Answer not found." nếu không tìm thấy mẫu trong văn bản.
    """
    match = re.search(pattern, text_response)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return "Answer not found."