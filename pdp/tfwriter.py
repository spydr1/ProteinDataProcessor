import os
import tensorflow as tf
from pdp.file_io import load_fasta
from pdp.data import Fulldata, SS8

from glob import glob
import tqdm
import random

class PretrainDataTFWriter:
    def __init__(self, path,
                 shuffle=True,
                 seed=12345):
        self.path = path
        glob_pattern = os.path.join(path, '*.fasta')
        self.file_list = glob(glob_pattern)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.file_list)


    def to_tfrecord(self,
                    tfrecord_file,
                    max_sequence_length = 512
                    ) -> None:
        """
        export to tfreocrd file.
        """
        dirname = os.path.dirname(tfrecord_file)
        os.makedirs(dirname, exist_ok=True)

        # todo : parallel ?
        record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file,record_option)
        for file in tqdm.tqdm(self.file_list):
            my_fasta = load_fasta(file)

            if len(my_fasta.seq)+2 < max_sequence_length:
                tfrecord_writer.write(my_fasta.serialize())
        tfrecord_writer.close()

from esm.data import ESMStructuralSplitDataset

class FullDataTFWriter:
    def __init__(self,
                 split_level='superfamily',
                 cv_partition='4',
                 split='train',
                 seed=12345,
                 shuffle=True):

        esm_structural_train = ESMStructuralSplitDataset(
            split_level='superfamily',
            cv_partition='4',
            split='train',
            root_path=os.path.expanduser('~/.cache/torch/data/esm'),
            download=True
        )
        if shuffle:
            random.seed(seed)
            random.shuffle(esm_structural_train.names)
        self.data = esm_structural_train


    def to_tfrecord(self,
                    tfrecord_file,
                    max_sequence_length = 512
                    ) -> None:
        """
        export to tfreocrd file.
        """
        # todo : parallel ?

        dirname = os.path.dirname(tfrecord_file)
        os.makedirs(dirname, exist_ok=True)

        record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file,record_option)
        for data in tqdm.tqdm(self.data):
            my_fulldata = Fulldata(seq=data['seq'],
                                   coords=data['coords'],
                                   ss8=SS8(data['ssp']),
                                   dist=data['dist'])

            if len(my_fulldata.seq)+2 < max_sequence_length:
                tfrecord_writer.write(my_fulldata.serialize())
        tfrecord_writer.close()

