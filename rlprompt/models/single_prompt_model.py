import torch
from typing import Optional, List, Union, Any, Dict
from .base_model import BaseModel


class SinglePromptModel(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        prompt_length: int,
        prompt_train_batch_size: int,
        prompt_infer_batch_size: int,
        source_str: str,
    ):
        super().__init__()
        self._model = model
        self.prompt_length = prompt_length
        self.prompt_train_batch_size = prompt_train_batch_size
        self.prompt_infer_batch_size = prompt_infer_batch_size
        self.source_str = source_str

    def _get_prompt_source(self, batch_size: int) -> List[str]:
        return [self.source_str for _ in range(batch_size)]

    def generate(
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        num_beams: Optional[int],
        max_new_tokens: Optional[int] = None,
        infer: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if infer: 
            batch_size = min(self.prompt_infer_batch_size, len(source_texts))
        else: 
            batch_size = self.prompt_train_batch_size
        prompt_source = self._get_prompt_source(batch_size=batch_size)

        if max_new_tokens is None: 
            max_new_tokens = self.prompt_length
        return self._model.generate(source_texts=prompt_source,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    num_beams=num_beams,
                                    max_new_tokens=max_new_tokens,
                                    **kwargs)

    def teacher_forcing(
        self,
        source_texts: List[str],
        sample_ids: torch.LongTensor,
        **kwargs
    ) -> Dict[str, Any]:
        prompt_source = self._get_prompt_source(self.prompt_train_batch_size)
        return self._model.teacher_forcing(source_texts=prompt_source,
                                           sample_ids=sample_ids,
                                           **kwargs)
