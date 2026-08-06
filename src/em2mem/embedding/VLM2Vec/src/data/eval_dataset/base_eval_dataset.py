from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Dataset, Features, Value, Sequence


# Schema for evaluation dataset, not used in the code.
EVAL_QRY_FEATURES = Features(**{
    "query_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "query_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "cand_text": Sequence(Value(dtype='string')),  # Here it's only for hard negatives.
    "cand_video": Sequence({
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    }),
    "dataset_infos": {
        "cand_names": Sequence(Value(dtype='string')),
        "label_name": Value(dtype='string')
    }
})

EVAL_CAND_FEATURES = Features(**{
    "cand_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "cand_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "dataset_infos": {
        "cand_name": Value(dtype='string'),
    },
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class ImageVideoInstance:
    """
    len(bytes) == len(path) == len(resolution) == 1: image
    len(bytes) == len(path) == len(resolution) > 1: multi-image / video
    """
    def __init__(self, bytes, paths, resolutions):
        assert len(bytes) == len(paths) == len(resolutions)
        self.bytes = bytes
        self.paths = paths
        self.resolutions = resolutions

    def to_dict(self):
        return {
            "bytes": self.bytes,
            "paths": self.paths,
            "resolutions": self.resolutions,
        }


class AutoEvalPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoEvalPairDataset.registry:
            AutoEvalPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                # print(f"Adding {dataset_name}")
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass


def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data.get('query_text', batch_data.get('cand_text', [])))
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data

    return wrapper


def generate_cand_dataset(dataset, corpus):
    """
    Used for generating candidate datasets.
    Flatten candidates, merge with corpus, deduplication
    """
    cand_rows = []
    all_cand_name = set()
    for row in dataset:
        assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"])
        for cand_text, cand_image, cand_name in zip(row["cand_text"], row["cand_image"], row["dataset_infos"]["cand_names"]):
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_text": [cand_text],
                    "cand_image": [cand_image],
                    "dataset_infos": {"cand_name": cand_name},
                })
                all_cand_name.add(cand_name)

    if corpus is not None:
        for row in corpus:
            assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"]) == 1
            cand_name = row["dataset_infos"]["cand_names"][0]
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_text": row["cand_text"],
                    "cand_image": row["cand_image"],
                    "dataset_infos": {"cand_name": row["dataset_infos"]["cand_names"][0]},
                })
                all_cand_name.add(cand_name)

    cand_dataset = Dataset.from_list(cand_rows)
    return cand_dataset
