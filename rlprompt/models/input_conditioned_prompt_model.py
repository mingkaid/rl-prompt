import torch
from typing import Optional, List, Union, Any, Dict
from .base_model import BaseModel


class InputConditionedPromptModel(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        prompt_length: int,
        source_train_reps: int,
        source_infer_reps: int
    ):
        super().__init__()
        self._model = model
        self.prompt_length = prompt_length
        self.source_train_reps = source_train_reps
        self.source_infer_reps = source_infer_reps

    def _do_source_reps(
        self, 
        source_texts: List[str], 
        num_reps: int
    ) -> List[str]:
        source_reps = []
        for text in source_texts: 
            for _ in range(num_reps): 
                source_reps.append(text)
        return source_reps

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
        if max_new_tokens is None: 
            max_new_tokens = self.prompt_length
        if infer: 
            num_reps = self.source_infer_reps
        else: 
            num_reps = self.source_train_reps
        source_reps = self._do_source_reps(source_texts, num_reps)
        # print(source_reps)
        return self._model.generate(source_texts=source_reps,
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
        source_reps = self._do_source_reps(source_texts, self.source_train_reps)
        # print(sample_ids)
        return self._model.teacher_forcing(source_texts=source_reps,
                                           sample_ids=sample_ids,
                                           **kwargs)
