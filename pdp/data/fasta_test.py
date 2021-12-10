import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import logging

import copy
from pdp.data.fasta import Fasta, load_fasta
from pdp.data.feature import AminoAcid


class TestStringMethods(tf.test.TestCase):
    def setUp(self):
        self.data = Fasta("1NAC_A", "MAC")

    def test_serialize(self):
        serialized_data = self.data.serialize()

        features = {
            "fasta": tf.io.FixedLenFeature([], tf.string),
            "seq": tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
        }
        eg = tf.io.parse_single_example(serialized_data, features)
        self.assertEqual(self.data.get_seq(), AminoAcid(eg["seq"].numpy()))

    def test_export_fasta(self):
        path = os.path.join(self.get_temp_dir(), "test.fasta")
        self.data.to_fasta(path)

        loaded_fasta = load_fasta(path)
        self.assertEqual(loaded_fasta.get_seq(), self.data.get_seq())


if __name__ == "__main__":
    tf.test.main()
