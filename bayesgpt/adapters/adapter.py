import numpy as np


class Adapter:

    def __init__(self):
        super().__init__()
        pass

    def convert_dtype(self):
        raise NotImplementedError

    def atleast_2d(self):
        raise NotImplementedError

    def concatenate(self):
        raise NotImplementedError
