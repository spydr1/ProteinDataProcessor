from pdp.data.feature import AminoAcid
from pdp.data.utils import _bytes_feature, _int64_feature
import tensorflow as tf
import os


class Fasta:
    # todo : 정의, 구성.

    """
    desc:
    """

    def __init__(self, name: str, seq: AminoAcid):
        self.name = name
        self.seq = seq if isinstance(seq, AminoAcid) else AminoAcid(seq)
        self._idx = None

    def __repr__(self):
        return f"({self.name}, {self.seq})"

    def __eq__(self, other):
        if isinstance(other, Fasta):
            if other.name == self.name and other.seq == self.seq:
                return True
        return False

    def get_seq(self):
        return self.seq

    def get_name(self):
        return self.name

    @property
    def idx(self):
        """
        get the index number of amino acid.
        """

        if not self._idx:
            self._idx = self.seq.get_idx()
        return self._idx

    def serialize(self) -> bytes:
        """
        get the serialized data.
        """

        name = self.name.encode("ascii")
        seq_idx = self.seq.get_idx()
        feature = {
            "fasta": _bytes_feature([name]),
            "seq": _int64_feature(seq_idx),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def to_fasta(self, fasta_file):
        """
        export to fasta file.
        """
        with open(fasta_file, mode="w") as file_obj:
            file_obj.write(f">{self.name}\n")
            file_obj.write(self.seq)


def load_fasta(fasta_file):
    basename = os.path.basename(fasta_file)
    fasta_name = os.path.splitext(basename)[0]

    with open(fasta_file) as file_obj:
        _ = file_obj.readline().strip()
        seq = file_obj.readline().strip()

    new_fasta = Fasta(fasta_name, seq)
    return new_fasta
