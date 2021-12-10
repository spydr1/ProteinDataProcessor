import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import logging

import copy
from pdp.data.feature import AminoAcid, SS3, SS8


class TestStringMethods(tf.test.TestCase):
    def setUp(self):
        self.raw_aa = "MAC"
        self.raw_aa_idx = [0, 1, 2]

        self.raw_ss3 = "HHHCCC"
        self.raw_ss3_idx = [0, 1, 2, 2, 3]

        self.raw_ss8 = "HECTSGBIX"
        self.raw_ss8_idx = [0, 1, 2, 3, 4, 5, 6, 7]

    def test_AminoAcid(self):
        self.assertEqual(self.raw_aa, AminoAcid(self.raw_aa))

    def test_AminoAcid_get_idx(self):
        aa = AminoAcid(self.raw_aa_idx)
        self.assertEqual(self.raw_aa_idx, aa.get_idx())

    def test_SS3(self):
        self.assertEqual(self.raw_ss3, SS3(self.raw_ss3))

    def test_SS3_get_idx(self):
        ss3 = SS3(self.raw_ss3_idx)
        self.assertEqual(self.raw_ss3_idx, ss3.get_idx())

    def test_SS8(self):
        self.assertEqual(self.raw_ss8, SS8(self.raw_ss8))

    def test_SS8_get_idx(self):
        ss8 = SS8(self.raw_ss8_idx)
        self.assertEqual(self.raw_ss8_idx, ss8.get_idx())


if __name__ == "__main__":
    tf.test.main()
