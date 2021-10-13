

# length
# 논문에서는 어떻게 썼는가 ?
# split
# -> tfdata
import logging
from pdp.data import Fasta
import tensorflow as tf
import os
import random
from typing import List
import tqdm

# todo : shuffle, uniref50이 이미 잘 섞여있다면 필요 없을 텐데.
class PretrainWriter :
    def __init__(self,
                 filepath : str ,
                 max_sequence_length:int = 1024,
                 buffer_size:int = 1000000):
        self.filepath = filepath
        self.max_sequence_length = max_sequence_length
        self.buffer_size = buffer_size

    def _filtering(self,test_mode) -> List[Fasta]:

        count = 0
        fasta_list = []
        current_seq = ''

        with open(self.filepath) as fileobject:
            for idx, line in enumerate(fileobject):

                # remove the \n symbol
                line = line.strip()
                # file ex)
                # >UniRef50_A0A1E3NP16 ~~
                # MAB ~~~

                # todo : 이렇게 끝내면 마지막 하나는 안담기네 ..?
                if line[0] == '>':
                    # ex) >UniRef50_K7G060 Titin n=25 Tax=Amniota TaxID=32524 RepID=K7G060_PELSI
                    # <cls>, <eos> 토큰 추가되니까 +2
                    if  idx > 0 and len(current_seq)+2<self.max_sequence_length:
                        print(f"index : {idx} {count}", end='\r')

                        _fasta = [current_name,current_seq]
                        fasta_list.append(_fasta)
                        count += 1

                    current_name = line.split(' ')[0][1:] # name ex) >UniRef50_A0A1E3NP16 blah blah~~
                    current_seq =''
                else:
                    current_seq+=line

                # test
                if test_mode and count ==1000 :
                    break

        return fasta_list

    def to_tfrecord(self, rewrite=False, test_mode=False, seed=12345) -> None:
        # todo : 병렬 처리.
        if rewrite :
            import shutil
            _path = '~/.cache/tfdata/pretrain/'
            _path = os.path.expanduser(_path)
            shutil.rmtree(_path,ignore_errors=True)


        tfrecord_path = f'~/.cache/tfdata/pretrain/0.tfrecord'
        tfrecord_path = os.path.expanduser(tfrecord_path)
        dirname = os.path.dirname(tfrecord_path)
        if os.path.exists(tfrecord_path):
            raise OSError("already exists")
        os.makedirs(dirname, exist_ok=True)

        fasta_list = self._filtering(test_mode=test_mode)
        random.seed(seed)
        random.shuffle(fasta_list)

        record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path, record_option)

        for idx, _fasta in enumerate(fasta_list):
            _fasta = Fasta(_fasta[0],_fasta[1])
            tfrecord_writer.write(_fasta.serialize())

            if (idx+1) % self.buffer_size == 0:
                logging.info("buffer is full.")
                tfrecord_writer.close()

                if test_mode and (idx + 1) == 1000:
                    break
                logging.info("new tfrecord file.")
                tfrecord_path = f'~/.cache/tfdata/pretrain/{(idx+1) // self.buffer_size}.tfrecord'
                tfrecord_path = os.path.expanduser(tfrecord_path)
                tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path, record_option)
        tfrecord_writer.close()


writer = PretrainWriter('/Pharmcadd/uniref50.fasta')
writer.to_tfrecord(rewrite=True,test_mode=False)
