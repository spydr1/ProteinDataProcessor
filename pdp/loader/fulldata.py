import os

import tensorflow as tf

# import tensorflow.experimental.numpy as np
import logging
from pdp.utils.vocab import aa_idx_vocab

import numpy as np

# @tf.function()
# def get_random_ij(shape=[2], minval=1):
#
#     def _get_random_ij(maxval):
#         tensor = tf.random.uniform(
#             shape=[2],
#             minval=1,
#             maxval=tf.maximum(length + 1 - patch_size, 1),
#             # dtype=tf.dtypes.int32,
#         )
#         return
#
#     return _get_random_ij

# new span
#
# todo : max_sequence_length -> seq_length , config와 동일하게
# todo : loader output 원래 데이터랑 비교할수 있게끔 역변환 함수 추가해야함.
class FullDataLoader:
    """
    dataloader for fine-tuning,

    Attributes:
        split_level : 1-letter amino acid.
        coords : coordination of Ca
        dist : distance map
        ss8 : 8-class secondary structure
        ss3 : 3-class secondary structure

    """

    def __init__(
        self,
        split_level="superfamily",
        cv_partition: int = 4,
    ):
        """
        Args :
            split_level : string.
                how to clustering the pdb, it comes from facebook-esm.
                you can choose one of superfamily, family and fold.
            cv_partition : int
                for cross validation, you have to set partition of data.
            seed : int
                for data robustness,
        """

        self.split_level = split_level
        self.cv_partition = str(cv_partition)

    def _check_exist(self) -> bool:
        """
        check the existence of tfdata.
        """
        tfrecord_path = "~/.cache/tfdata/fulldata/"
        self.tfrecord_path = os.path.expanduser(tfrecord_path)

        return os.path.exists(self.tfrecord_path)

    # todo :
    def _download(self):
        """
        downloading the tfdata.
        """
        import urllib.request

        url = "https://drive.google.com/drive/folders/12hwbZwdwUYNaenUqJL7ybyHqsbNP0_C_?usp=sharing"
        urllib.request.urlretrieve(url, os.path.expanduser("~/.cache/tfdata/fulldata"))

    # todo : tfx, data versioning -> ML OPS에 대한 생각.
    # todo : tfdata로 return되는게 편한걸까 ?
    def __call__(
        self,
        file=None,
        mode="train",
        is_training: bool = False,
        max_sequence_length: int = 1024,
        buffer_size: int = 200,
        global_batch_size: int = 8,
        bins=16,
        triu_k=6,
        contact_k=4,
        seed=12345,
        patch_size=256,
    ) -> tf.data.TFRecordDataset:
        """

        Args :
            file : string.
                name of tfdata. (default: ~ )
            mode : string.
                train or valid.
                in train, data will be repeating, shuffling.
                in valid, no repeat and shuffle.
            is_training : boolean.
                in true, data will be repeating, shuffling.
                in false, no repeat and shuffle.
            max_sequence_length :
            buffer_size :
            global_batch_size :
            bins :

        Return :
            TFRecordDataset
            It contains the fulldata information.
        """
        # tfrecordfile을 찾는 것 + 읽는 것 두개 합쳐져 있는 것 같은데 ?

        if self._check_exist() is False:
            os.makedirs(self.tfrecord_path)
            raise Exception(
                "Download is not available. It will be added. Please use esm2tfdata() written in pdp/writer/fulldata_test.py"
            )
            # raise OSError("file is not exist.")
        # else :
        #     pass
        # self._download()

        # todo : 함수로 빼자
        if mode == "train":
            _partition_list = ["0", "1", "2", "3", "4"]
            tfrecord_files = []
            desc = ""
            for _partition in _partition_list:
                if self.cv_partition != _partition:
                    desc += f"{_partition} "
                    _path = os.path.expanduser(
                        f"~/.cache/tfdata/fulldata/{self.split_level}/{_partition}.tfrecord"
                    )
                    tfrecord_files.append(_path)
                logging.info(f"partiion list : {desc}")

        else:
            tfrecord_files = os.path.expanduser(
                f"~/.cache/tfdata/fulldata/{self.split_level}/{self.cv_partition}.tfrecord"
            )

        if file:
            tfrecord_files = file

        features = {
            "fasta": tf.io.FixedLenFeature([], tf.string),
            "seq": tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
            "dist": tf.io.FixedLenFeature([], tf.string),
            "coords": tf.io.FixedLenFeature([], tf.string),
            "ss8": tf.io.RaggedFeature(value_key="ss8", dtype=tf.int64),
            "ss_weight": tf.io.RaggedFeature(value_key="ss_weight", dtype=tf.int64),
        }

        # todo : without tf.deivece func, it's using the gpu memory for loading data.
        with tf.device("cpu"):
            dataset = tf.data.TFRecordDataset(
                tfrecord_files,
                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                compression_type="GZIP",
            )

        if is_training is True:
            dataset = dataset.shuffle(buffer_size, seed=seed)
            dataset = dataset.repeat()
        else:
            logging.info("you are using dataset for evaluation. no repeat, no shuffle")

        triangle_mat = np.triu(
            np.ones([max_sequence_length, max_sequence_length]), k=triu_k
        )
        triangle_mat = triangle_mat + triangle_mat.T

        contact_mat = np.triu(
            np.ones([max_sequence_length, max_sequence_length]), k=contact_k
        )
        contact_mat = contact_mat + contact_mat.T

        # todo : k=3 정도로 ..?

        def _parse_function(example_proto):
            eg = tf.io.parse_single_example(example_proto, features)
            fasta = eg["fasta"]
            seq = eg["seq"]
            # length = len(eg["seq"])
            length = tf.size(eg["seq"], out_type=tf.dtypes.int32)
            # todo : pre-train과 똑같은 input을 만들기 위해서 cls, eos 추가.. 넣는게 맞는걸까 ?
            seq = tf.concat(
                [[aa_idx_vocab["<cls>"]], seq, [aa_idx_vocab["<eos>"]]], axis=0
            )
            seq_mask = tf.ones(length + 2, dtype=tf.int64)
            # seq_mask = tf.ones(length, dtype=tf.int64)
            # seq_mask = tf.pad(seq_mask, [[1,1]])

            # set the zero weight to <cls> and <eos>
            ss8 = eg["ss8"]
            ss8 = tf.pad(ss8, [[1, 1]])

            ss_weight = eg["ss_weight"]
            ss_weight = tf.pad(
                ss_weight, [[1, 1]]
            )  # equal - ss_weight = tf.concat([[0], ss_weight, [0]], axis=0)

            # distance
            dist = tf.io.parse_tensor(eg["dist"], out_type=tf.float32)

            # mistake :
            # tf.math.is_nan(nan_test) == False
            # tf.math.is_nan(nan_test) is False
            nan_mask = tf.math.is_nan(dist) == [False]

            if contact_k > 0:
                diagonal_mask = tf.constant(contact_mat, tf.bool)[:length, :length]
                dist_mask = nan_mask & diagonal_mask
            else:
                dist_mask = nan_mask

            # contact_aa
            # dist_only_long = tf.where(dist_mask, 16 - dist, 0)
            # contact_pair = tf.gather(eg["seq"], tf.math.top_k(dist_only_long).indices[:,0])
            # contact_pair = tf.pad(contact_pair,[[1, 1]])

            # distance range is 2~18 -> clipping range is 0~16
            # todo : distance, nan 값 등장에 대한 분석
            dist = tf.math.multiply_no_nan(
                dist, tf.cast(dist_mask, tf.float32)
            )  # todo : nan 을 그냥 곱하면 안되는거 .. 정리하자.
            dist = tf.floor(dist)
            dist = tf.clip_by_value(dist - 2, 0, bins - 1)

            # zero padding to <cls> and <eos>
            dist_mask = tf.pad(dist_mask, [[1, 1], [1, 1]])

            # is it contacted ?
            # contact = tf.cast(dist<=6,tf.int32)
            # contact_mask = (
            #     tf.constant(contact_mat, tf.int32)[:length, :length] * nan_mask
            # )
            # contact = tf.reduce_any(tf.equal(contact*contact_mask,1),axis=-1)
            # contact = tf.pad(contact,[[1, 1]])

            # zero padding to <cls> and <eos>
            dist = tf.pad(dist, [[1, 1], [1, 1]])
            # todo : think about "dtype"

            # ij = np.random.randint(1, max(length + 1 - patch_size, 1), size=2)
            ij = tf.random.uniform(
                shape=[2],
                minval=1,
                maxval=tf.maximum(
                    length + 2 - patch_size, 2
                ),  # minval <= val < maxval <- not include maxval
                dtype=tf.dtypes.int32,
            )
            i, j = ij[0], ij[1]

            return {
                "input_fasta": fasta,
                "input_seq": tf.cast(seq, tf.int32),
                "input_seq_mask": tf.cast(seq_mask, tf.int32),
                "input_ss8_target": tf.cast(ss8, tf.int32),
                "input_ss_weight": tf.cast(ss_weight, tf.int32),
                "input_dist_target": tf.cast(dist, tf.int32),
                "input_dist_mask": tf.cast(dist_mask, tf.int32),
                "input_patch_dist_target": tf.cast(
                    dist[i : i + patch_size, j : j + patch_size], tf.int32
                ),
                "input_patch_dist_mask": tf.cast(
                    dist_mask[i : i + patch_size, j : j + patch_size],
                    tf.int32,
                ),
                "input_ij": tf.cast(ij, tf.int32),
                "input_length": length,
            }

        padded_shapes = {
            "input_fasta": [],
            "input_seq": [max_sequence_length],
            "input_seq_mask": [max_sequence_length],
            "input_ss8_target": [max_sequence_length],
            "input_ss_weight": [max_sequence_length],
            "input_dist_target": [max_sequence_length, max_sequence_length],
            "input_dist_mask": [max_sequence_length, max_sequence_length],
            "input_ij": [2],
            "input_length": [],
            "input_patch_dist_target": [patch_size, patch_size],
            "input_patch_dist_mask": [patch_size, patch_size],
        }
        zero = tf.constant(0, dtype=tf.int32)
        padded_value = {
            "input_fasta": "",
            "input_seq": zero,
            "input_seq_mask": zero,
            "input_ss8_target": zero,
            "input_ss_weight": zero,
            "input_dist_target": zero,
            "input_dist_mask": zero,
            "input_ij": zero,
            "input_length": zero,
            "input_patch_dist_target": zero,
            "input_patch_dist_mask": zero,
        }

        dataset = dataset.map(
            _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.padded_batch(
            global_batch_size, padded_shapes=padded_shapes, padding_values=padded_value
        )

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        # 1. pre-processing, parsing.
