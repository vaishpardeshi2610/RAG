import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF file using PyMuPDF."""
    text_content = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            text_content.append(text)
    return text_content

pdf_path = "attention_is.pdf"  # Replace with your PDF file path
documents = extract_text_from_pdf(pdf_path)
