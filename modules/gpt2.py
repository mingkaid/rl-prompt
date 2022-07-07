import torch
import texar.torch as tx
from collections import defaultdict
from typing import Tuple, Dict, List, Optional


from sql.utils import colorful_warning
from sql.types import FloatTensor, LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPT2Model(object):
    MAX_SEQ_LENGTH = 128
    MODEL_NAME = "gpt2-small"
    CACHE_DIR = "/export/share/Experiments/20210517/gpt2-models/"
    CKPT_PATHS = {
        "snli": "/export/share/Experiments/20210518/gpt2_snli/model.ckpt",
        "multinli": "/export/share/Experiments/20210518/gpt2_multinli/model.ckpt",
    }

    def __init__(
            self,
            task_name: str,
            batch_size: int = 32,
    ) -> None:
        if task_name not in self.CKPT_PATHS.keys():
            raise ValueError

        tokenizer = tx.data.GPT2Tokenizer(
            pretrained_model_name=self.MODEL_NAME,
            cache_dir=self.CACHE_DIR)

        model = tx.modules.GPT2Decoder(
            pretrained_model_name=self.MODEL_NAME,
            cache_dir=self.CACHE_DIR)

        checkpoint_path = self.CKPT_PATHS[task_name]
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            colorful_warning(f"Loaded checkpoint from {checkpoint_path}", bg="blue")

        model.eval()
        model.to(device)
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size

        # self._max_decoding_length = 128
        # if max_decoding_length > model.hparams.position_size:
        #     raise ValueError("max_decoding_length should not "
        #                      "be greater than position size")

    def process_examples_into_batch(self, examples: List[str]) -> Dict[str, LongTensor]:
        features = defaultdict(list)
        for example in examples:
            text_ids, length = self._tokenizer.encode_text(
                text=example,
                max_seq_length=self.MAX_SEQ_LENGTH,
                append_eos_token=True)

            features["text_ids"].append(text_ids)
            features["length"].append(length)

        batch = {}
        for key, values in features.items():
            batch[key] = torch.tensor(values).to(device)

        return batch

    @torch.no_grad()
    def batch_forward(
            self,
            batch: Dict[str, LongTensor]
    ) -> Tuple[FloatTensor, tx.modules.TransformerDecoderOutput, int]:

        input_ids = batch["text_ids"]
        batch_size = input_ids.size()[0]
        outputs = self._model(inputs=input_ids)
        losses = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=batch["text_ids"][:, 1:],
            logits=outputs.logits[:, :-1, :],
            sequence_length=batch["length"] - 1,
            average_across_batch=False,
            average_across_timesteps=True,
            sum_over_batch=False,
            sum_over_timesteps=False)

        return losses, outputs, batch_size

    def forward(self, sentences: List[str]) -> FloatTensor:
        losses = []
        for index in range(0, len(sentences), self._batch_size):
            i_0 = index
            i_1 = index + self._batch_size
            batch = self.process_examples_into_batch(sentences[i_0: i_1])
            batch_losses, _, _ = self.batch_forward(batch)
            losses.append(batch_losses)

        return torch.cat(losses, dim=0)
