

# vocab
# mlm

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import logging

from pdp.loader.fulldata import FullDataLoader
from pdp.writer.fulldata import FulldataWriter
from glob import glob
import numpy as np
from pdp.data.feature import AminoAcid
from pdp.data.fulldata import Fulldata
from esm.data import ESMStructuralSplitDataset

import copy
class TestStringMethods(tf.test.TestCase):
    def setUp(self):
        self.log= logging.getLogger( "TrainDataTest" )
        esm_structural_train = ESMStructuralSplitDataset(
            split_level='superfamily',
            cv_partition='4',
            split='train',
            root_path=os.path.expanduser('~/.cache/torch/data/esm'),
            download=True
        )

        self.test_data = esm_structural_train[0]
        # todo : self.unk_fasta=['>unk~~', 'MABABBBBBBBBBBBBBB'] ... self.correct_fasta=[] ..


    def test_TrainDataValidation(self):

        batch_size = 1
        obj_data_loader = FullDataLoader()
        train_data = obj_data_loader(is_training=False,
                                     sequence_length=1024,
                                     batch_size=batch_size,
                                     buffer_size=1)

        for idx, d in enumerate(train_data):
            _length = list(map(sum,d["input_seq_mask"].numpy()))
            _seq = d["input_seq"].numpy()

            tfdata_to_seq = list(map(lambda x,l : AminoAcid(x[1:l-1]), _seq, _length)) # todo : element is not Aminoacid.
        self.assertEqual(AminoAcid(tfdata_to_seq[0]),AminoAcid(self.test_data["seq"]),"input sequence is different from ground truth") # batch size




    # def setUp(self) -> None:
    #     # tfrecord
    #     self.tempdir = self.create_tempdir()
    #     self.tfrecord_file = os.path.join(self.tempdir, 'test.tfrecords')
    #
    #     self.test_data = open(os.path.join(self.tempdir,"test.fasta"),mode='w')
    #     for _bet in ['A','B','C']:
    #         self.test_data.write(f'2N64_{_bet}')
    #         self.test_data.write(f'MACACA{_bet}')
    #     self.test_data.close()
    #
    #     with open(self.tempdir,"test.fasta", mode='r') as fileobj:
    #         print(fileobj.readlines())
    #
    #
    #     self.fasta = Fasta(self.fasta_name,self.seq)


    # def test_LoadFasta(self):
    #     # when i read fasta_file, it must same to original.
    #     self.fasta_file = os.path.join(self.tempdir, f"{self.fasta_name}.fasta")
    #     self.fasta.to_fasta(self.fasta_file)
    #     self.assertEqual(load_fasta(self.fasta_file), self.fasta,
    #                      msg=f'loaded fasta : {load_fasta(self.fasta_file)} \n'
    #                          f'original fasta : {self.fasta}')
#
#



import sys
if __name__ == '__main__' :
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TrainDataTest").setLevel(logging.DEBUG)
    tf.test.main()
