import tenseal as ts
from torch import Tensor
from typing import Dict
from flib.utils import convert_size


class SEAL(object):
    def __init__(self, context: ts.Context) -> None:
        self.context = context

    def encrypt(
        self, weight: Dict[str, Tensor], batch: bool = True
    ) -> Dict[str, ts.CKKSTensor]:
        size = 0
        w_dict = dict()
        for key, value in weight.items():
            w_dict[key] = ts.ckks_tensor(self.context, value, batch=batch)
            size += len(w_dict[key].serialize())
        print("[OK] Encypted weight, size:", convert_size(size))
        return w_dict

    def decrypt(self, weight: Dict[str, ts.CKKSTensor]) -> Dict[str, Tensor]:
        w_dict = dict()
        for key, value in weight.items():
            w_dict[key] = Tensor(value.decrypt(self.context.secret_key()).tolist())
        return w_dict

    def serialize(self, weight: Dict[str, ts.CKKSTensor]) -> Dict[str, bytes]:
        w_dict = dict()
        for key, value in weight.items():
            w_dict[key] = value.serialize()
        return w_dict

    def deserialize(self, weight: Dict[str, bytes]) -> Dict[str, ts.CKKSTensor]:
        w_dict = dict()
        for key, value in weight.items():
            w_dict[key] = ts.ckks_tensor_from(self.context, value)
        return w_dict
