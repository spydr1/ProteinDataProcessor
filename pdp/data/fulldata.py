from pdp.data.feature import AminoAcid, SS8
from pdp.data.utils import _bytes_feature, _int64_feature
import tensorflow as tf
from Bio.PDB.vectors import Vector
from typing import Tuple
from pdp.utils import vocab

# from typing import Sequence, Tuple, List, Text
import numpy as np


class Fulldata:
    """
    for fine-tuning,

    Attributes:
        aa : 1-letter amino acid.
        coords : coordination of Ca
        dist : distance map
        ss8 : 8-class secondary structure
        ss3 : 3-class secondary structure

    """

    def __init__(self, aa: AminoAcid, coords: np.ndarray, dist: Tuple, ss8: SS8):
        self.aa = aa if isinstance(aa, AminoAcid) else AminoAcid(aa)
        self.coords = coords
        self.dist = dist
        self.ss8 = ss8 if isinstance(ss8, SS8) else SS8(ss8)
        self.ss3 = self.ss8.get_ss3()

    def __repr__(self):
        return (
            f"seq : {self.aa}\n"
            f"coords : {self.coords}\n"
            f"dist : {self.dist}\n"
            f"ss8 : {self.ss8}"
        )

    def serialize(self) -> bytes:
        """
        get the serialized data.
        """
        # todo (complete) : convert_to_tensor 바꿔야함, int64 -> int32 로 바꾸는게 생각보다 시간이 오래걸리는것 같다.? 정확하게는 모르겠고.
        seq_idx = self.aa.idx
        ss3_idx = self.ss3.idx
        ss8_idx = self.ss8.idx

        ss_weight = [
            0 if idx == vocab.ss3_idx_vocab["X"] else 1 for idx in ss8_idx
        ]  # "X" means unknown.

        # dist & coords is list. so it is need to serialize.
        dist = tf.io.serialize_tensor(self.dist)
        coords = tf.io.serialize_tensor(self.coords)
        # todo : np.tobytes() 확인 해보자 -> 값은 다르던데..
        feature = {
            "seq": _int64_feature(seq_idx),
            "ss3": _int64_feature(ss3_idx),
            "ss8": _int64_feature(ss8_idx),
            "dist": _bytes_feature([dist.numpy()]),  # todo .numpy() 확인해보기
            "coords": _bytes_feature([coords.numpy()]),
            "ss_weight": _int64_feature(ss_weight),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def to_dict(self, pickle_file):
        import pickle

        data = {
            "seq": self.aa,
            "ss8": self.ss8,
            "dist": self.dist,
            "coords": self.coords,
        }
        with open(pickle_file, "wb") as fw:
            pickle.dump(data, fw)


def load_fulldata(fulldata_file):
    import pickle

    with open(fulldata_file, "rb") as fr:
        data_loaded = pickle.load(fr)
        new_fulldata = Fulldata(
            data_loaded["seq"],
            data_loaded["coords"],
            data_loaded["dist"],
            data_loaded["ss8"],
        )

    return new_fulldata
