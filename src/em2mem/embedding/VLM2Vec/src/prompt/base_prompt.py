from abc import ABCMeta, abstractmethod


class AutoPrompt(metaclass=ABCMeta):
    # Base class for auto prompt.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoPrompt.registry:
            AutoPrompt.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, prompt_family, task_name=None, task_type=None, *args, **kwargs):
        try:
            return cls.registry[prompt_family](task_name, task_type, *args, **kwargs)
        except Exception as e:
            if prompt_family not in cls.registry:
                f"Unknown prompt_family: {prompt_family}"
            raise e

    @classmethod
    def register(cls, prompt_family):
        def inner_wrapper(wrapped_class):
            if prompt_family in cls.registry:
                print(f"[Alert] AutoPrompt: a class in the same name ({prompt_family}) has been registered")
            else:
                cls.registry[prompt_family] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass
