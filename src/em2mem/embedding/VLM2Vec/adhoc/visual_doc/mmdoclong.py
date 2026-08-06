from datasets import load_dataset
import fitz  # PyMuPDF
from PIL import Image
import os
import json
import base64
from io import BytesIO
import ast

# Load dataset
dataset = load_dataset("yubo2333/MMLongBench-Doc")["train"]

# Directory containing PDFs
pdf_dir = "/fsx/sfr/data/MMEB/Visual_Doc/mmlongbench/documents"

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Dictionary to store images
all_images = {}
processed_pdfs = {}
pdf_corpus_mapping = {}  # Mapping from pdf_file_name to base corpus_id
existing_corpus_ids = set()  # Track already added corpus-ids

queries = []
corpus = []
qrels = []
corpus_counter = 0

# Process each PDF
for qid, doc in enumerate(dataset):
    pdf_file_name = doc["doc_id"]
    pdf_path = os.path.join(pdf_dir, pdf_file_name)

    if doc['evidence_pages'] == []:
        continue

    # Ensure the file exists before processing
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file {pdf_file_name} not found. Skipping.")
        continue

    if pdf_file_name not in processed_pdfs:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        images = []

        # Convert each page to an image
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        processed_pdfs[pdf_file_name] = images
        pdf_corpus_mapping[pdf_file_name] = corpus_counter
        corpus_counter += len(images)  # Increment by number of images
    else:
        images = processed_pdfs[pdf_file_name]

    # Ensure pdf_file_name is in pdf_corpus_mapping before access
    if pdf_file_name not in pdf_corpus_mapping:
        print(f"Error: {pdf_file_name} not found in pdf_corpus_mapping. Skipping.")
        continue

    base_corpus_id = pdf_corpus_mapping[pdf_file_name]
    all_images[pdf_file_name] = images

    try:
        evidence_pages = ast.literal_eval(doc['evidence_pages'])
        if not isinstance(evidence_pages, list):
            raise ValueError("Invalid evidence pages format")
    except Exception as e:
        print(f"Error parsing evidence pages for {pdf_file_name}: {e}")
        continue

    if len(evidence_pages) == 0:
        continue
    queries.append({
        "query-id": qid,
        "query": doc["question"],
        "corpus_range": list(range(base_corpus_id, base_corpus_id + len(images)))
    })

    for page_number in evidence_pages:
        qrels.append({
            'query-id': qid,
            'corpus-id': base_corpus_id + int(page_number),
            'score': 1
        })

    # Store encoded images in corpus if not already added
    for img_id, image in enumerate(images):
        corpus_id = base_corpus_id + img_id  # Fix corpus ID numbering
        if corpus_id not in existing_corpus_ids:
            corpus.append({
                "corpus-id": corpus_id,
                "image": encode_image(image)
            })
            existing_corpus_ids.add(corpus_id)

# Function to save data in JSONL format
def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

print('size of qrels:', len(qrels))
print('size of queries:', len(queries))
print('size of corpus:', len(corpus))

save_dir = "/fsx/sfr/data/MMEB/Visual_Doc/mmlongbench/test/"
os.makedirs(save_dir, exist_ok=True)
queries_file = "queries.jsonl"
corpus_file = "corpus.jsonl"
qrels_file = "qrels.jsonl"

# Save to JSONL
save_jsonl(os.path.join(save_dir, queries_file), queries)
save_jsonl(os.path.join(save_dir, corpus_file), corpus)
save_jsonl(os.path.join(save_dir, qrels_file), qrels)
