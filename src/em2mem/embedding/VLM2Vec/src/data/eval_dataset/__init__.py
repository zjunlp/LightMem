# Video Classification
from .video_classification_datasets import load_video_class_dataset
from .ssv2_dataset import load_ssv2_dataset

# Video QA
from .videomme_dataset import load_videomme_dataset
from .mvbench_dataset import load_mvbench_dataset
from .nextqa_dataset import load_nextqa_dataset
from .egoschema_dataset import load_egoschema_dataset
from .activitynetqa_dataset import load_activitynetqa_dataset
from .videommmu_dataset import load_videommmu_dataset

# Video Retrieval
from .msrvtt_dataset import load_msrvtt_dataset
from .didemo_dataset import load_didemo_dataset
from .msvd_dataset import load_msvd_dataset
from .youcook2_dataset import load_youcook2_dataset
from .vatex_dataset import load_vatex_dataset

from .gui_dataset import load_gui_dataset

# Temporal Grounding
from .moment_retrieval_datasets import load_moment_retrieval_dataset
from .momentseeker_dataset import load_momentseeker_dataset

# MMEB
from .image_cls_dataset import load_image_cls_dataset
from .image_qa_dataset import load_image_qa_dataset
from .image_t2i_eval import load_image_t2i_dataset
from .image_i2t_eval import load_image_i2t_dataset
from .image_i2i_vg_dataset import load_image_i2i_vg_dataset

# VisDoc
from .vidore_dataset import load_vidore_dataset
from .visrag_dataset import load_visrag_dataset
