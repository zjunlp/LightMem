from __future__ import annotations
from .prompts import get_prompt 
from string import Template
import inspect
from abc import ABC, abstractmethod
from .backends import get_interface_for_inference
from tqdm import tqdm 
import time 
import os 
from typing import (
    Any, 
    List, 
    Optional, 
    Dict, 
)

class NonCachedLLMOperator(ABC): 

    def __init__(
        self, 
        prompt_name: str, 
        model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        params = inspect.signature(self._preprocess).parameters 
        for param in params:
            if not param.endswith("_list"):
                raise ValueError(
                    "The name of each argument in the `preprocess` function must end with `_list`."
                )
        
        self.set_prompt(prompt_name)
        if model_name is not None:
            self._interface = get_interface_for_inference(model_name, **kwargs)
        else:
            self._interface = None

        self._model_name = model_name if model_name is not None else "Anonymous Model"
        # If the name of model is a directory path, 
        # we use the name of the directory as the model name
        if os.path.isdir(self._model_name):
            self._model_name = os.path.basename(self._model_name)

    def _check_prompt_identifiers(self) -> bool:
        params = inspect.signature(self._preprocess).parameters
        return all(
            f"{identifier}_list" in params
            for identifier in self._prompt.get_identifiers()
        )

    @abstractmethod
    def _preprocess(self, *args) -> List[List[Dict[str, str]]]:
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _aggregate(self, responses: List[Dict[str, Any]]) -> Any:
        return responses 
    
    def _check(self) -> bool:
        return self._interface is not None
    
    def set_prompt(self, prompt: str | Template) -> None:
        if isinstance(prompt, Template):
            self._prompt = prompt
        else:
            self._prompt = get_prompt(prompt)
        if not self._check_prompt_identifiers():
            raise ValueError(
                "The prompt identifiers are not consistent with the arguments of the `preprocess` function."
            )
    
    def from_operator(self, operator: NonCachedLLMOperator) -> None:
        """Copy the attributes from another operator."""
        self.set_prompt(operator._prompt)
        self._interface = operator._interface
        self._model_name = operator._model_name
    
    def __call__(
        self, 
        *args, 
        batch_size: int = 1, 
        aggregate: bool = True, 
        **kwargs
    ) -> Any:
        """Generate text using the LLM."""
        if not self._check():
            raise ValueError("The `interface` is not set.")
        
        messages_list = self._preprocess(*args)
        size = len(messages_list)

        progress_bar = tqdm(
            total=size, 
            desc=f"{self._model_name} model is used", 
        )
        final_responses = []
        
        for i in range(0, size, batch_size):
            batch_messages_list = [
                messages_list[batch_indice] 
                for batch_indice in range(i, min(i + batch_size, size))
            ]
            results = self._interface(batch_messages_list, **kwargs)
            if isinstance(results, dict):
                results = [results]
            for result in results:
                final_responses.append(result)
                progress_bar.update(1)
                # To make the progress bar update more smoothly
                time.sleep(0.1)

        progress_bar.close()
        if aggregate:
            return self._aggregate(final_responses)
        return final_responses 