import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from typing import Sequence, Tuple, List, Text


from Bio.PDB.vectors import Vector
from pdp.vocab import aa_idx_vocab, ss3_idx_vocab, ss8_idx_vocab, ss8_ss3_vocab

import tensorflow as tf

# Pretrain
FastaNameFormat = Text
FastaSeqFormat = Text
FastaFormat = Tuple[FastaNameFormat,FastaSeqFormat]
MSAFormat = Sequence[FastaFormat]

# Full
SS3Format = List[Text]
SS8Foramt = List[Text]
CoordsFormat = Sequence[Vector]
DistFormat = List[Tuple]


DataFormat = Tuple[FastaNameFormat, MSAFormat, CoordsFormat]

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
        return aa_idx(self)

    # todo : except ? , what is proper that when i check the validation ? in __init__ ?, I don't want to waste computation.

    def vocab_check(self) -> None:
        for idx, _seq in enumerate(self) :
            assert _seq in aa_idx_vocab, f"\n" \
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

class SS3(str) :
    def get_idx(self):
        return ss3_idx(self)

    def vocab_check(self) -> None:
        for idx, _seq in enumerate(self) :
            assert _seq in ss3_idx_vocab, f"\n" \
                                         f"\"{_seq}\" is not existed in vocab. \n" \
                                         f"check your sequence \"{self}\" \n" \
                                         f"idx number : {idx+1}"

class SS8(str) :
    def get_ss3(self)->SS3:
        return SS3(''.join([ss8_ss3_vocab[_seq] for _seq in self]))

    def get_idx(self):
        return ss8_idx(self)

    def vocab_check(self) -> None:
        for idx, _seq in enumerate(self) :
            assert _seq in ss8_idx_vocab, f"\n" \
                                         f"\"{_seq}\" is not existed in vocab. \n" \
                                         f"check your sequence \"{self}\" \n" \
                                         f"idx number : {idx+1}"



class Fulldata:
    def __init__(self, seq:AminoAcid, coords:Vector, dist:DistFormat, ss8:SS8):
        self.seq = seq if isinstance(seq, AminoAcid) else AminoAcid(seq)
        self.coords = coords
        self.dist =dist
        self.ss8 = ss8 if isinstance(ss8, SS8) else SS8(ss8)
        self.ss3 = ss8.get_ss3()
        self._aaidx = None
        self._ss8idx = None
        self._ss3idx = None


    def __repr__(self):
        return f"(seq : {self.seq} coords : {self.coords} dist : {self.dist} ss8 {self.ss8})"

    @property
    def aaidx(self):
        if not self._aaidx:
            self._aaidx = self.seq.get_idx()
        return self._aaidx

    @property
    def ss3idx(self):
        if not self._ss3idx:
            self._ss3idx = self.ss3.get_idx()
        return self._ss3idx

    @property
    def ss8idx(self):
        if not self._ss8idx:
            self._ss8idx = self.ss8.get_idx()
        return self._ss8idx

    def serialize(self) -> bytes:
        """
        get the serialized data.
        """
        seq_idx = tf.convert_to_tensor(self.seq.get_idx(), dtype=tf.int32)

        ss3_idx = tf.convert_to_tensor(self.ss3.get_idx(), dtype=tf.int32)
        ss8_idx = tf.convert_to_tensor(self.ss8.get_idx(), dtype=tf.int32)
        ss_weights = [0 if idx==8 else 1 for idx in self.ss8.get_idx()]

        dist = tf.convert_to_tensor(self.dist, dtype=tf.float32)
        dist = tf.io.serialize_tensor(dist)

        coords = tf.convert_to_tensor(self.coords, dtype=tf.float32)
        coords = tf.io.serialize_tensor(coords)


        feature = {
            'seq': _int64_feature(seq_idx),
            'ss3': _int64_feature(ss3_idx),
            'ss8': _int64_feature(ss8_idx),
            'dist': _bytes_feature([dist.numpy()]),
            'coords': _bytes_feature([coords.numpy()]),
            'ss_weights' : _int64_feature(ss_weights)
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def aa_idx(seq : AminoAcid) -> List[int]:
    """
    convert the amino acid to index.
    ex ) MAC -> [11,1,3]
    """
    return [aa_idx_vocab[_seq] for _seq in seq]


def ss3_idx(seq : AminoAcid) -> List[int]:
    return [ss3_idx_vocab[_seq] for _seq in seq]

def ss8_idx(seq : AminoAcid) -> List[int]:
    return [ss8_idx_vocab[_seq] for _seq in seq]


# todo : gap, Is gap is unknown ? length, mask for MLM, exception gap
# todo : if aa is gap, mask or not ?

