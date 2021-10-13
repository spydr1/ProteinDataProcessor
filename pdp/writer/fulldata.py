import logging
import os
import tensorflow as tf
import tqdm

from esm.data import ESMStructuralSplitDataset
from pdp.data import Fulldata, SS8


class FulldataWriter:
    def __init__(self,
                 split_level='superfamily',
                 cv_partition='4',
                 split='train',
                 seed=12345,
                 shuffle=True):

        esm_structural_train = ESMStructuralSplitDataset(
            split_level=split_level,
            cv_partition=cv_partition,
            split=split,
            root_path=os.path.expanduser('~/.cache/torch/data/esm'),
            download=True
        )
        self.split_level =split_level
        self.cv_partition = cv_partition
        self.split = split
        # if shuffle:
        #     random.seed(seed)
        #     random.shuffle(esm_structural_train.names)

        self.data = esm_structural_train


    def to_tfrecord(self,
                    max_sequence_length = 512
                    ) -> None:
        """
        export to tfreocrd file.
        """

        tfrecord_file = f'~/.cache/tfdata/{self.split_level}/{self.cv_partition}.tfrecord'
        tfrecord_file = os.path.expanduser(tfrecord_file)
        dirname = os.path.dirname(tfrecord_file)

        if not os.path.exists(tfrecord_file):
            raise OSError("already exists")
        os.makedirs(dirname, exist_ok=True)
        # record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        # tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file,record_option)
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file)

        # todo : is it possible to parallel processing ?
        # todo : angles ?
        for data in tqdm.tqdm(self.data):
            my_fulldata = Fulldata(seq=data['seq'],
                                   coords=data['coords'],
                                   ss8=SS8(data['ssp']),
                                   dist=data['dist'])
            # seq has two additional token, '<cls>' ,'<eos>'
            if len(my_fulldata.seq)+2 < max_sequence_length:
                tfrecord_writer.write(my_fulldata.serialize())
        tfrecord_writer.close()


def esm2tfdata() -> None:

    for split_level in ['family', 'superfamily', 'fold']:
        for cv_partition in ['0', '1', '2', '3', '4']:
            logging.info(f'{split_level} {cv_partition}')
            print(f'{split_level} {cv_partition}')
            writer = FulldataWriter(split_level=split_level, cv_partition=cv_partition, split='valid')
            writer.to_tfrecord()
