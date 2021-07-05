import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from typing import Sequence, Tuple, List, Text


from Bio.PDB.vectors import Vector
from pdp.vocab import aa2idx_vocab

import tensorflow as tf

FastaNameFormat = Text
FastaSeqFormat = Text
FastaFormat = Tuple[FastaNameFormat,FastaSeqFormat]
MSAFormat = Sequence[FastaFormat]
CoordsFormat = Sequence[Vector]
DataFormat = Tuple[FastaNameFormat, MSAFormat, CoordsFormat]

PretrainVocab = aa2idx_vocab
# pretrain dataset
# load fasta
# seq <-> one hot

# train dataset
# seq, msa, distmap, vector
# deserialized -> serialized
# msa -> one hot
# distance map

# prediction
# seq, msa, distmap


class AminoAcid(str):
    def get_idx(self):
        return seq2idx(self)

    # todo : except ? , what is proper that when i check the validation ? in __init__ ?, I don't want to waste computation.

    def vocab_check(self) -> None:
        for idx, _seq in enumerate(self) :
            assert _seq in aa2idx_vocab, f"\n" \
                                         f"\"{_seq}\" is not existed in vocab. \n" \
                                         f"check your sequence \"{self}\" \n" \
                                         f"idx number : {idx+1}"

class FastaName(str):
    def __init__(self, value : str):
        assert '_' in value, "pleas set the name format \"name_chain\" "
        self = value # todo : is it right method?


class Fasta:
    def __init__(self, name:FastaName, seq:AminoAcid):
        self.name = name if isinstance(name, FastaName) else FastaName(name)
        self.seq = seq if isinstance(seq, AminoAcid) else AminoAcid(seq)
        self._idx = None

    def __repr__(self):
        return f"({self.name}, {self.seq})"

    def __eq__(self, other):
        if isinstance(other,Fasta):
            if other.name == self.name and other.seq == self.seq:
                return True
        return False

    @property
    def idx(self):
        """
        get the index number of amino acid.
        Index number is declared int aa2idx_vocab.
        """
        if not self._idx:
            self._idx = self.seq.get_idx()
        return self._idx

    def serialize(self) -> bytes:
        """
        get the serialized data.
        """

        name = self.name.encode('ascii')
        seq_idx = tf.convert_to_tensor(self.seq.get_idx(), dtype=tf.int32)
        feature = {
            'fasta': _bytes_feature([name]),
            'seq': _int64_feature(seq_idx),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def to_fasta(self,fasta_file):
        """
        export to fasta file.
        """
        with open(fasta_file,mode='w') as file_obj:
            file_obj.write(f">{self.name}\n")
            file_obj.write(self.seq)




def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def seq2idx(seq : AminoAcid) -> List[int]:
    """
    convert the amino acid to index.

    ex ) MAC -> [11,1,3]
    """
    return [aa2idx_vocab[_seq] for _seq in seq]



# todo : gap, Is gap is unknown ? length, mask for MLM, exception gap
# todo : if aa is gap, mask or not ?

