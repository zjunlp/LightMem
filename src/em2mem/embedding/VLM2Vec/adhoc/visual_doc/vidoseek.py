from datasets import load_dataset
import fitz  # PyMuPDF
from PIL import Image
import os
import json
import base64
from io import BytesIO

# Load dataset
file_path = "/fsx/sfr/data/MMEB/Visual_Doc/ViDoSeek/vidoseek.json"
with open(file_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


pdf_dir = "/fsx/sfr/data/MMEB/Visual_Doc/ViDoSeek/vidoseek_pdf_document"

all_images = {}
processed_pdfs = {}
pdf_corpus_mapping = {}  # Mapping from pdf_file_name to base corpus_id
existing_corpus_ids = set()  # Track already added corpus-ids

queries = []
corpus = []
qrels = []
corpus_counter = 0

# Process each PDF
for qid, doc in enumerate(dataset['examples']):
    pdf_file_name = doc["meta_info"]['file_name']
    pdf_path = os.path.join(pdf_dir, pdf_file_name)

    if doc['meta_info']['reference_page'] == []:
        continue

    if pdf_file_name not in processed_pdfs:
        # Check if the file exists before reading
        if os.path.exists(pdf_path):
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
            corpus_counter += len(images)
    else:
        images = processed_pdfs[pdf_file_name]

    base_corpus_id = pdf_corpus_mapping[pdf_file_name]
    all_images[pdf_file_name] = images

    queries.append({
        "query-id": qid,
        "query": doc["query"],
        "corpus_range": list(range(base_corpus_id, base_corpus_id + len(images)))
    })

    # Assign qrels for pages in the same PDF (score = 2)
    for img_id, _ in enumerate(images):
        qrels.append({
            'query-id': qid,
            'corpus-id': base_corpus_id + img_id,
            'score': 2
        })

    # Assign qrels for reference pages (score = 3)
    for page_number in doc['meta_info']['reference_page']:
        qrels.append({
            'query-id': qid,
            'corpus-id': base_corpus_id + int(page_number),
            'score': 3
        })

    # Store encoded images in corpus if not already added
    for img_id, image in enumerate(images):
        corpus_id = base_corpus_id + img_id
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

print('size of qrels', len(qrels))
print('size of queries', len(queries))
print('size of corpus', len(corpus))

save_dir = "/fsx/sfr/data/MMEB/Visual_Doc/ViDoSeek/test/"
os.makedirs(save_dir, exist_ok=True)
queries_file = "queries.jsonl"
corpus_file = "corpus.jsonl"
qrels_file = "qrels.jsonl"

# Save to JSONL
save_jsonl(save_dir + queries_file, queries)
save_jsonl(save_dir + corpus_file, corpus)
save_jsonl(save_dir + qrels_file, qrels)