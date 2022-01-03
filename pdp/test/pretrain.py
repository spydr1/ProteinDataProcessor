

# vocab
# mlm

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import logging

from pdp.loader.pretraindata import PretrainDataLoader
from pdp.writer.pretraindata import PretrainDataWriter
from glob import glob
import numpy as np
from pdp.data.feature import AminoAcid

import copy
class TestStringMethods(tf.test.TestCase):
    def setUp(self):
        self.log= logging.getLogger( "TrainDataTest" )
        # todo : self.unk_fasta=['>unk~~', 'MABABBBBBBBBBBBBBB'] ... self.correct_fasta=[] ..


    def test_LoadFastaFile(self):
        # save 한거 load한거 비교
        pass

    def create_FastaFile(self, case=["A", "C", "D"]) :
        # todo: 새로운 test에 맞춰서 이거 수정해야함.
        dir_FastaFile = os.path.join(self.get_temp_dir(), f"test{''.join(case)}.fasta")
        obj_fasta_file = open(dir_FastaFile, mode="w")
        self.seq_list = []

        for _bet in case:
            obj_fasta_file.write(f">2N64_{_bet}\n")
            seq = f"MACACA{_bet}"
            self.seq_list.append(seq)
            obj_fasta_file.write(seq+"\n")
        obj_fasta_file.close()
        return dir_FastaFile

    def test_TrainDataValidation(self):
        # load 한것 = input으로 만든 것 비교

        dir_correct_fasta_file = self.create_FastaFile()
        tf.print("\n--- example of fasta file ---")
        with open (dir_correct_fasta_file,mode="r") as fileobj :
            for line in fileobj.readlines():
                tf.print(line.strip())

        obj_pretrain_data_writer = PretrainDataWriter(filepath = dir_correct_fasta_file)
        dir_pretrain_data_dir = os.path.join(self.get_temp_dir(),"pretrain")
        obj_pretrain_data_writer.to_tfrecord(tfrecord_dir = dir_pretrain_data_dir,
                                             test_mode = True,
                                             shuffle = False,)

        tfrecord_path = os.path.expanduser(os.path.join(dir_pretrain_data_dir,"*.tfrecord"))
        tfrecord_list = glob(tfrecord_path)

        batch_size = 3
        tf.print(f"\n--- tfreocrd list ---\n{tfrecord_list}")
        obj_data_loader = PretrainDataLoader(tfrecord_list)
        train_data = obj_data_loader(is_training=False,
                                     max_sequence_length=1024,
                                     batch_size=batch_size,
                                     buffer_size=1)

        origin_seq = [AminoAcid(_seq) for _seq in self.seq_list]
        for idx, d in enumerate(train_data):
            tfdata_to_seq = invert_seq(d)
            self.assertEqual(origin_seq, tfdata_to_seq,"input sequence is different from ground truth")

        def test_TrainData_unk_case(self):
            # unk AA입력 됬을 때
            dir_unk_fasta_file = self.create_FastaFile()

        # todo : 1. train data로 들어가는 sequence 2.masking 답으로 바꿔서 fasta file에서와 train data가 같은지.

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


def invert_seq(data:tf.Tensor):

    length = data["length"].numpy()
    input_seq = data["input_seq"].numpy()
    lm_pos = data["input_lm_positions"].numpy()
    lm_target = data["input_lm_target"].numpy()
    lm_len = np.array(list(map(lambda x: sum(x.numpy()),data["input_lm_weights"])))

    aa_list = []
    for _seq,_pos,_target,_lm_len,_length in zip(input_seq,lm_pos,lm_target,lm_len,length):
        _seq[_pos[:_lm_len]]=_target[:_lm_len] # length is dynamic.
        aa_list.append(AminoAcid(_seq[1:_length+1])) # exclude <cls>, <eos>
    return aa_list



import sys
if __name__ == '__main__' :
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TrainDataTest").setLevel(logging.DEBUG)
    tf.test.main()
