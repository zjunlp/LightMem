from openai import OpenAI 
import warnings 
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams 
from typing import (
    Optional,  
    Dict, 
    Any, 
    Callable, 
    List, 
)

class OpenAIClient(OpenAI): 
    """
    A subclass of ``openai.OpenAI`` that wraps around the chat completions API to provide a simple interface 
    for generating text with the OpenAI language models, along with cost estimation.

    This class provides functionality for interacting with OpenAI's chat models, allowing for 
    customized generation with options like temperature, top_p, streaming. 
    It also supports post-processing of the generated content and retrying the request up to a 
    specified tolerance level if any errors occur during the API call.
    """

    def get_text_generation_output(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4o", 
        post_processor: Optional[Callable[[str], Any]] = None,
        max_tolerance: int = 3,
        temperature: float = 0.75, 
        top_p: float = 0.95,
        stream: bool = True, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the OpenAI chat completions API and return the response content, 
        with optional post-processing.
        """
        response_content = None 
        counter = 0 
        content = ''
        reasoning_content = None 

        while response_content is None and counter <= max_tolerance:
            try:
                response = self.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                    **kwargs
                )
                if stream:
                    chunks = []
                    reasoning_chunks = [] 
                    for chunk in response:
                        if len(chunk.choices) > 0:
                            chunks.append(chunk.choices[0].delta.content or '')
                            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                                reasoning_chunks.append(chunk.choices[0].delta.reasoning_content or '')
                        else:
                            warnings.warn(
                                "Find a chunk without `choices` attribute. "
                                "The model may reject to answer the question. "
                                "Please check the question and the model you use.",
                                UserWarning
                            )
                    content = ''.join(chunks)
                    if len(reasoning_chunks) > 0:
                        reasoning_content = ''.join(reasoning_chunks)
                else:
                    content = response.choices[0].message.content
                    if hasattr(response.choices[0].message, "reasoning_content"):
                        reasoning_content = response.choices[0].message.reasoning_content 
            except Exception as e:
                print(e)
            finally: 
                response_content = content if post_processor is None else post_processor(content)
                counter += 1
        
        outputs = {
            "content": content, 
            "processed_content": response_content,
        }
        if reasoning_content is not None:
            outputs["reasoning_content"] = reasoning_content

        return outputs

def openai_api_batch_inference(
    clients: List[OpenAIClient], 
    messages_list: List[List[Dict[str, str]]], 
    model: str = "gpt-4o", 
    post_processor: Optional[Callable[[str], Any]] = None,
    max_tolerance: int = 3,
    temperature: float = 0.75, 
    top_p: float = 0.95,
    stream: bool = True, 
    **kwargs
) -> List[Dict[str, Any]]: 
    """Process multiple OpenAI API requests in parallel using thread pool."""
    n_jobs = len(clients) 
    if len(messages_list) != n_jobs:
        raise ValueError(f"The number of clients ({n_jobs}) must match the number of messages ({len(messages_list)}).")
    
    apply_func = lambda client, messages: client.get_text_generation_output(
        messages, 
        model, 
        post_processor, 
        max_tolerance, 
        temperature, 
        top_p, 
        stream, 
        **kwargs, 
    )

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(apply_func, clients[i], messages_list[i])
            for i in range(n_jobs)
        ]
        results = [future.result() for future in futures]

    return results

class NativeLLMClient: 
    """A client for native LLM inference. Powered by vLLM."""

    def __init__(self, model: str, **kwargs) -> None:
        self.model = LLM(model=model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __call__(
        self, 
        messages_list: List[List[Dict[str, str]]], 
        post_processor: Optional[Callable[[str], Any]] = None, 
        enable_thinking: Optional[bool] = None, 
        **kwargs
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Generate text using the native LLM."""
        if enable_thinking is not None:
            texts = self.tokenizer.apply_chat_template(
                messages_list, 
                tokenize=False, 
                add_generation_prompt=True,  
                enable_thinking=enable_thinking, 
            )
        else:
            texts = self.tokenizer.apply_chat_template(
                messages_list, 
                tokenize=False, 
                # Add additional tokens to ensure the chat model 
                # Generate a system response instead of continuing a users message
                add_generation_prompt=True,  
            )

        if len(kwargs) > 0:
            sampling_params = SamplingParams(**kwargs)
            outputs = self.model.generate(texts, sampling_params)
        else: 
            # Use default sampling params recommended by the model creator
            outputs = self.model.generate(texts)

        new_outputs = [] 
        for output in outputs: 
            content = output.outputs[0].text 
            processed_content = content 
            if post_processor is not None:
                processed_content = post_processor(content)
            if isinstance(processed_content, dict):
                new_outputs.append(
                    {
                        "content": content, 
                        **processed_content, 
                    }
                )
            else:   
                new_outputs.append(
                    {
                        "content": content, 
                        "processed_content": processed_content, 
                    }
                )

        return new_outputs if len(new_outputs) > 1 else new_outputs[0]
    
class OpenAIClientPool: 
    """A pool of OpenAI clients for batch inference."""

    def __init__(
        self, 
        api_keys: List[str] | str, 
        base_urls: List[str] | str, 
        model: str = "gpt-4o", 
        **kwargs
    ) -> None:
        """Initialize a pool of OpenAI clients."""
        if len(api_keys) != len(base_urls):
            raise ValueError(
                f"The number of api_key ({len(api_keys)}) must match the number of base_url ({len(base_urls)})."
            )
        
        self.client_pool = [
            OpenAIClient(
                api_key=api_key, 
                base_url=base_url, 
                **kwargs
            )
            for api_key, base_url in zip(api_keys, base_urls)
        ] 
        self.model = model  
    
    def __call__(
        self, 
        messages_list: List[List[Dict[str, str]]], 
        post_processor: Optional[Callable[[str], Any]] = None, 
        **kwargs
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Generate text using the OpenAI clients."""
        max_batch_size = len(self.client_pool) 

        outputs = [] 
        for i in range(0, len(messages_list), max_batch_size):
            batch_messages_list = messages_list[i: i + max_batch_size]
            batch_clients = self.client_pool[0: len(batch_messages_list)]
            if len(batch_clients) == 1:
                client = batch_clients[0]
                outputs.append(
                    client.get_text_generation_output(
                        batch_messages_list[0], 
                        model=self.model, 
                        post_processor=post_processor, 
                        **kwargs
                    )
                ) 
            else:
                outputs.extend(
                    openai_api_batch_inference(
                        batch_clients, 
                        batch_messages_list, 
                        model=self.model, 
                        post_processor=post_processor, 
                        **kwargs
                    )
                )

        return outputs if len(outputs) > 1 else outputs[0]
    
    @property
    def pool_size(self) -> int:
        return len(self.client_pool)

def get_interface_for_inference(
    model: str, 
    api_keys: Optional[List[str] | str] = None, 
    base_urls: Optional[List[str] | str] = None, 
    **kwargs
) -> OpenAIClientPool | NativeLLMClient:
    """Get an interface for inference."""
    if api_keys is not None and base_urls is not None:
        return OpenAIClientPool(api_keys, base_urls, model, **kwargs)
    if api_keys is not None or base_urls is not None:
        raise ValueError(
            "Either both `api_keys` and `base_urls` must be provided, or neither."
        )
    interface = NativeLLMClient(model, **kwargs) 
    return interface