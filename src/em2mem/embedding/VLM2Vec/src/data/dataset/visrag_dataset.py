from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes
from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS


def process_query(query, prompt, image_token=''):
    if prompt:
        query = f'{prompt} {query} {image_token}'
    else:
        query = f'{query} {image_token}'
    return query

query_source2prompt = {
    "NeurIPS Papers": "This query is about a research paper from NeurIPS, a leading AI/ML conference. The document contains technical discussions, methodologies, and findings. Identify relevant papers and sections that address the query: ",  # 10,000
    "Textbooks": "This query is related to a college-level textbook, which provides structured explanations, definitions, and examples. Find the most relevant concepts or explanations that address the query: ",    # 5,000
    "ICML Papers": "This query is about a research paper from ICML, a leading AI/ML conference. The document contains theoretical insights, experiments, and applications. Identify relevant papers and sections that best answer the query: ",    # 5,000
    "Manuallib": "This query pertains to a product manual, which contains detailed technical specifications, usage instructions, and troubleshooting steps. Find the most relevant section that answers the query: ",    # 20,000
    "ArxivQA": "This query is related to retrieving a relevant figure from an ArXiv research paper. The retrieved figure should contain scientific plots, mathematical visualizations, or experimental results that best address the query: ",  # 25,856
    "ChartQA": "This query is related to retrieving a relevant chart that visually represents numerical or categorical data. The retrieved chart should contain bar graphs, line charts, or other visual elements necessary to analyze trends, compare values, or extract insights related to the query: ",	  # 4,224
    "MP-DocVQA": "This query is related to retrieving a relevant page from a multi-page document, such as reports, invoices, or research papers. The retrieved document should contain text, tables, or structured information necessary to answer the query: ",	  # 10,624
    "InfoVQA": "This query is related to retrieving an infographic that visually presents statistical or factual information using charts, icons, and structured layouts. The retrieved image should contain the necessary visual elements to provide the best context for answering the query: ",	  # 17,664
    "PlotQA": "This query relates to retrieving a relevant plot or chart that visually represents numerical data. The retrieved figure should contain the necessary information to analyze trends, compare values, or extract key insights related to the query: ",	  # 56,192
    "SlideVQA": "This query is related to retrieving a relevant presentation slide that visually presents structured information. The retrieved slide should contain the necessary text, charts, or graphics to provide the best answer to the query: ",	  # 8,192
}
target_source2prompt = {
    "Textbooks": "A textbook page with structured educational content and explanations.",
    "ICML Papers": "A research paper from ICML, covering machine learning topics.",
    "NeurIPS Papers": "A research paper from NeurIPS on AI and ML topics.",
    "Manuallib": "A product manual page with technical specifications and instructions.",
    "InfoVQA": "An infographic with structured data, charts, and annotations.",
    "PlotQA": "A numerical data visualization, such as bar charts or line graphs.",
    "SlideVQA": "A presentation slide with text, bullet points, and diagrams.",
    "ArxivQA": "A figure from a research paper, including plots or experimental results.",
    "MP-DocVQA": "A page from a multi-page document with text or tables.",
    "ChartQA": "A statistical chart comparing values or analyzing trends.",
}

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    batch_size = len(batch_dict['query'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for query, image, source in zip(batch_dict['query'], batch_dict['image'], batch_dict['source']):
        # ignore prompt since most prompts too long
        query = process_query(query, prompt=query_source2prompt.get(source, ""), image_token="")
        query_texts.append(query)
        pos_text = process_query('', prompt=target_source2prompt.get(source, ""), image_token=VLM_IMAGE_TOKENS[model_backbone])
        pos_texts.append(pos_text)
        neg_texts.append("")
        if isinstance(image, Image.Image):
            # BC, datasets==2.21.0
            image_bytes = image_to_bytes(image)
            path = ""
        elif type(image) is dict:
            # datasets==3.3.2
            image_bytes = image['bytes']
            path = image['path']
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        query_images.append(None)
        pos_images.append({"bytes": [image_bytes], "paths": [path], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]})
        neg_images.append(None)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "visrag"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_visreg_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    dataset_split = kwargs.get("dataset_split", "train")
    global_dataset_name = kwargs.get("global_dataset_name", f'{DATASET_PARSER_NAME}/{dataset_name}')
    dataset_path = kwargs.get("dataset_path", None)

    if dataset_path:
        dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    elif dataset_name:
        dataset = load_dataset(dataset_name, split=dataset_split)


    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = global_dataset_name
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=128,
                          remove_columns=['image'],
                          # remove_columns=['query', 'image', 'source'],
                          drop_last_batch = True)
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    # print_master(f"Loaded {DATASET_PARSER_NAME}/{dataset_name} dataset with {num_rows} samples")
    return dataset
