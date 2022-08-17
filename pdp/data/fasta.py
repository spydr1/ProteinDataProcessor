from pdp.data.feature import AminoAcid
from pdp.data.utils import _bytes_feature, _int64_feature
import tensorflow as tf
import os


# todo : Fasta 를 위한
class Fasta:
    """
    Fasta data class

    Args:
        name: name of fasta
        seq: residue
    """

    def __init__(self, name: str, seq: AminoAcid):
        self.name = name
        self.aa = seq if isinstance(seq, AminoAcid) else AminoAcid(seq)

    def __repr__(self):
        return f"({self.name}, {self.aa})"

    def __eq__(self, other):
        if isinstance(other, Fasta):
            if other.name == self.name and other.aa == self.aa:
                return True
        return False

    def serialize(self) -> bytes:
        """
        get the serialized data.
        """

        name = self.name.encode("ascii")
        idx = self.aa.idx
        feature = {
            "fasta": _bytes_feature([name]),
            "seq": _int64_feature(idx),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # todo : biopython 을 쓸까 ?
    def to_fasta(self, fasta_file):
        """
        export to fasta file.
        """
        with open(fasta_file, mode="w") as file_obj:
            file_obj.write(f">{self.name}\n")
            file_obj.write(self.aa)


# todo : load 해야할 양식이 어떤것들이 있을까 ?  1. uniref 2. 단일 fasta 이것도 biopython을 쓸까? 단일이건 말건 > 기준으로 다 읽으면 되겠다. 원하는 index를 읽고 싶은 기능은 추가 할까 ?
def load_fasta(fasta_file):
    """
    get tha fasta.

    Args :
        fasta_file : file written fasta name and residue.
    """
    basename = os.path.basename(fasta_file)
    fasta_name = os.path.splitext(basename)[0]

    with open(fasta_file) as file_obj:
        _ = file_obj.readline().strip()
        seq = file_obj.readline().strip()

    new_fasta = Fasta(fasta_name, seq)
    return new_fasta
