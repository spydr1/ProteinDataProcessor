# length
# 논문에서는 어떻게 썼는가 ?
# split
# -> tfdata
import logging
from pdp.data.fasta import Fasta
import tensorflow as tf
import os
import random
from typing import List
import tqdm

# todo : shuffle, uniref50이 이미 잘 섞여있다면 필요 없을 텐데.
class PretrainDataWriter:
    def __init__(
        self, filepath: str, sequence_length: int = 1024, buffer_size: int = 100000
    ):
        """

        Args :
            filepath: directory of "uniref50.fasta" file, you can download from "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2018_03/uniref/"
            sequence_length:
            buffer_size:
        """
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size

    def _filtering(self, test_mode) -> List[Fasta]:

        count = 0
        fasta_list = []
        current_seq = ""
        current_name = ""

        with open(self.filepath) as fileobject:
            for idx, line in enumerate(fileobject):

                # remove the \n symbol
                line = line.strip()
                # file ex)
                # >UniRef50_A0A1E3NP16 ~~
                # MAB ~~~

                # todo (complete) : 이렇게 끝내면 마지막 하나는 안담기네 ..?
                if line[0] == ">":
                    # ex) >UniRef50_K7G060 Titin n=25 Tax=Amniota TaxID=32524 RepID=K7G060_PELSI
                    # <cls>, <eos> 토큰 추가되니까 +2
                    if idx > 0 and len(current_seq) + 2 < self.sequence_length:
                        print(f"index : {idx} {count}", end="\r")

                        _fasta = [current_name, current_seq]
                        fasta_list.append(_fasta)
                        count += 1

                    current_name = line.split(" ")[0][
                        1:
                    ]  # name ex) >UniRef50_A0A1E3NP16 blah blah~~
                    current_seq = ""
                else:
                    current_seq += line

                # test
                if test_mode and count == 1000:
                    break
            else:
                print(f"index : {idx} {count}", end="\r")

                _fasta = [current_name, current_seq]
                fasta_list.append(_fasta)
                count += 1
        return fasta_list

    def to_tfrecord(
        self,
        tfrecord_dir="~/.cache/tfdata/pretrain/",
        rewrite=False,
        test_mode=False,
        seed=12345,
        shuffle=True,
    ) -> None:
        # todo : 병렬 처리.

        if rewrite:
            import shutil

            tfrecord_dir = os.path.expanduser(tfrecord_dir)
            shutil.rmtree(tfrecord_dir, ignore_errors=True)

        tfrecord_path = os.path.join(tfrecord_dir, "0.tfrecord")
        tfrecord_path = os.path.expanduser(tfrecord_path)
        dirname = os.path.dirname(tfrecord_path)
        if os.path.exists(tfrecord_path):
            raise OSError("already exists")
        os.makedirs(dirname, exist_ok=True)

        fasta_list = self._filtering(test_mode=test_mode)

        if shuffle:
            random.seed(seed)
            random.shuffle(fasta_list)

        record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path, record_option)

        for idx, _fasta in enumerate(fasta_list):
            _fasta = Fasta(_fasta[0], _fasta[1])
            tfrecord_writer.write(_fasta.serialize())

            if (idx + 1) % self.buffer_size == 0:
                logging.info("buffer is full.")
                tfrecord_writer.close()

                if test_mode and (idx + 1) == 1000:
                    break
                logging.info("new tfrecord file.")
                tfrecord_path = (
                    f"~/.cache/tfdata/pretrain/{(idx+1) // self.buffer_size}.tfrecord"
                )
                tfrecord_path = os.path.expanduser(tfrecord_path)
                tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path, record_option)
        tfrecord_writer.close()
