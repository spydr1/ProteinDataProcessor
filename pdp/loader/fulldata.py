import os

import tensorflow as tf
import logging
from pdp.utils.vocab import aa_idx_vocab
import numpy as np

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
        seed: int = 12345,
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
        self.seed = seed

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
        sequence_length: int = 1024,
        buffer_size: int = 200,
        batch_size: int = 8,
        bins=16,
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
            sequence_length :
            buffer_size :
            batch_size :
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
            "seq": tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
            "dist": tf.io.FixedLenFeature([], tf.string),
            "coords": tf.io.FixedLenFeature([], tf.string),
            "ss8": tf.io.RaggedFeature(value_key="ss8", dtype=tf.int64),
            "ss_weight": tf.io.RaggedFeature(value_key="ss_weight", dtype=tf.int64),
        }

        dataset = tf.data.TFRecordDataset(
            tfrecord_files,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
            compression_type="GZIP",
        )

        if is_training is True:
            dataset = dataset.shuffle(buffer_size, seed=self.seed)
            dataset = dataset.repeat()
        else:
            logging.info("you are using dataset for evaluation. no repeat, no shuffle")

        triangle_mat = np.triu(np.ones([sequence_length, sequence_length]), k=1)
        # todo : k=3 정도로 ..?

        def _parse_function(example_proto):
            with tf.device("cpu"):
                eg = tf.io.parse_single_example(example_proto, features)

                seq = eg["seq"]
                length = len(eg["seq"])
                # todo : pre-train과 똑같은 input을 만들기 위해서 cls, eos 추가.. 넣는게 맞는걸까 ?
                seq = tf.concat(
                    [[aa_idx_vocab["<cls>"]], seq, [aa_idx_vocab["<eos>"]]], axis=0
                )
                seq_mask = tf.ones(length + 2, dtype=tf.int64)

                ss8 = eg["ss8"]

                # cls에 해당하는 부분 0으로 넣고, weight 0으로 만들기.
                ss8 = tf.concat([[0], ss8], axis=0)
                ss_weight = eg["ss_weight"]
                ss_weight = tf.concat([[0], ss_weight], axis=0)

                # distance
                dist = tf.io.parse_tensor(eg["dist"], out_type=tf.float32)
                nan_mask = tf.cast(tf.logical_not(tf.math.is_nan(dist)), tf.int64)
                dist = tf.cast(tf.floor(dist), dtype=tf.int64) * nan_mask
                # cls에 해당하는 부분 0으로 채워넣기.
                dist = tf.pad(dist, [[1, 0], [1, 0]])

                # distance range is 2~18
                # clipping range is 0~16
                # todo : distance, nan 값 등장에 대한 분석
                dist = tf.clip_by_value(dist, 2, bins + 1) - 2

                # distance mask
                dist_mask = (
                    tf.constant(triangle_mat, tf.int64)[:length, :length] * nan_mask
                )
                # cls에 해당하는 부분 0으로 채워넣기.
                dist_mask = tf.pad(dist_mask, [[1, 0], [1, 0]])

            return {
                "input_seq": tf.cast(seq, tf.int32),
                "input_seq_mask": tf.cast(seq_mask, tf.int32),
                "input_ss8_target": tf.cast(ss8, tf.int32),
                "input_ss_weight": tf.cast(ss_weight, tf.int32),
                "input_dist_target": tf.cast(dist, tf.int32),
                "input_dist_mask": tf.cast(dist_mask, tf.int32),
            }

        padded_shapes = {
            "input_seq": [sequence_length],
            "input_seq_mask": [sequence_length],
            "input_ss8_target": [sequence_length],
            "input_ss_weight": [sequence_length],
            "input_dist_target": [sequence_length, sequence_length],
            "input_dist_mask": [sequence_length, sequence_length],
        }
        zero = tf.constant(0, dtype=tf.int32)
        padded_value = {
            "input_seq": zero,
            "input_seq_mask": zero,
            "input_ss8_target": zero,
            "input_ss_weight": zero,
            "input_dist_target": zero,
            "input_dist_mask": zero,
        }

        dataset = dataset.map(
            _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=padded_shapes, padding_values=padded_value
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        # 1. pre-processing, parsing.
