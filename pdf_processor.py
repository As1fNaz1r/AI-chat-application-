# import PyPDF2
# from io import BytesIO
# from sentence_transformers import SentenceTransformer

# def process_pdfs(pdf_contents):
#     knowledge_base = []
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#     for pdf_content in pdf_contents:
#         pdf_file = BytesIO(pdf_content)
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
        
#         # Split text into chunks
#         chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
#         for chunk in chunks:
#             embedding = model.encode(chunk)
#             knowledge_base.append({"text": chunk, "embedding": embedding})

#     return knowledge_base

import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer

def process_pdfs(pdf_contents):
    knowledge_base = []
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')  # Force CPU usage

    for pdf_content in pdf_contents:
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]

        for chunk in chunks:
            embedding = model.encode(chunk, device='cpu')  # Force CPU usage
            knowledge_base.append({"text": chunk, "embedding": embedding})

    return knowledge_base
