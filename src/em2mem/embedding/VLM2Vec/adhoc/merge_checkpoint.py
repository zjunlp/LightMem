from src.arguments import ModelArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor


def main():
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    model_args: ModelArguments

    model = MMEBModel.build(model_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, "model_backbone", model_backbone)
    # processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args, is_trainable=False)
    model.config.save_pretrained(f'{model_args.model_name}/full_model/', safe_serialization=False)
    processor = load_processor(model_args)
    processor.save_pretrained(f'{model_args.model_name}/full_model/', safe_serialization=False)
    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained(f'{model_args.model_name}/full_model/', safe_serialization=False)


if __name__ == "__main__":
    main()
