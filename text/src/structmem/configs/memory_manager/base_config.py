from typing import Dict, Optional, Union, List


class BaseMemoryManagerConfig:
    """
    Config for LLMs.
    """
    def __init__(
        self,
        # General parameters
        model: Optional[Union[str, Dict]] = None,
        seed: int = 42,
        do_sample: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        stop: List[str] = [],
        # Other parameters
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        # Local model specific parameters
        host: Optional[str] = "http://localhost:11434",
        num_gpu: Optional[int] = -1, # number of GPUs to use, -1 means all available GPUs, 0 means CPU only
        main_gpu: Optional[int] = 0,
        gpu_memory_utilization: Optional[float] = 0.9,  # fraction of GPU memory to use
        trust_remote_code: bool = True,
        # API model specific parameters
        api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None, # OpenAI specific
        deepseek_base_url: Optional[str] = None, # DeepSeek specific
        vllm_base_url: Optional[str] = None, # vLLM specific
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ):

        # General parameters
        self.model = model
        self.seed = seed
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop

        # Other parameters
        self.enable_vision = enable_vision
        self.vision_details = vision_details

        # Local model specific parameters
        self.host = host
        self.num_gpu = num_gpu
        self.main_gpu = main_gpu
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code

        # API model specific parameters
        self.api_key = api_key
        self.openai_base_url = openai_base_url
        self.deepseek_base_url = deepseek_base_url
        self.vllm_base_url = vllm_base_url
        self.site_url = site_url
        self.app_name = app_name
