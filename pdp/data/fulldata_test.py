import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import logging

import copy
from pdp.data.fulldata import Fulldata, load_fulldata
from pdp.data.feature import AminoAcid
import numpy as np

np.zeros([3, 3])


class TestStringMethods(tf.test.TestCase):
    def setUp(self):
        coords = np.random.random([3, 3])
        dist = np.zeros([3, 3])
        dist[0, 1] = 1
        dist[1, 1] = 1
        dist[0, 2] = 3

        self.data = Fulldata("MAC", coords, dist, "HHH")

    def test_serialize(self):
        serialized_data = self.data.serialize()

        features = {
            "seq": tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
            "dist": tf.io.FixedLenFeature([], tf.string),
            "coords": tf.io.FixedLenFeature([], tf.string),
            "ss8": tf.io.RaggedFeature(value_key="ss8", dtype=tf.int64),
        }
        eg = tf.io.parse_single_example(serialized_data, features)
        self.assertEqual(self.data.aa, AminoAcid(eg["seq"].numpy()))

    def test_export_fulldata(self):
        path = os.path.join(self.get_temp_dir(), "test.fasta")
        self.data.to_dict(path)

        loaded_fulldata = load_fulldata(path)
        self.assertEqual(loaded_fulldata.aa, self.data.aa)


if __name__ == "__main__":
    tf.test.main()
