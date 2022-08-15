from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def greedy_search(self, *args, **kwargs):
        raise NotImplementedError

    def teacher_forcing(self, *args, **kwargs):
        raise NotImplementedError
