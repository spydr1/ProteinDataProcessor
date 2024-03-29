import numpy as np
import os
from glob import glob
import tensorflow as tf
import logging
from pdp.utils.vocab import aa_idx_vocab

# new span
#
import random

# todo : data split 에 대해 생각. 어떻게 데이터 버전 관리할지
# todo : tfx, data versioning -> ML OPS에 대한 생각.


class PretrainDataLoader:
    def __init__(self, tfrecord_path="~/.cache/tfdata/pretrain/*", seed: int = 12345):
        tfrecord_path = os.path.expanduser(tfrecord_path)
        self.files = sorted(glob(tfrecord_path))
        print(f"training files : {self.files}")
        self.seed = seed
        random.seed(seed)
        random.shuffle(self.files)

        self._train_len = round(len(self.files) * 0.9)
        # todo : numpy, tfrecord 버전 추가

    def download(self):
        pass

    def load(self, span: int) -> tf.data.TFRecordDataset:
        pass
        # 1. pre-processing, parsing.

    def __call__(
        self,
        mode="train",
        is_training: bool = False,
        max_sequence_length: int = 1024,
        num_token_predictions: int = 128,
        mask_ratio: float = 0.15,
        buffer_size: int = 200,
        batch_size: int = 8,
    ) -> tf.data.TFRecordDataset:
        """
        Args :
            mode: train or valid.
            is_training: boolean, if it is true, no shuffle, no repeat.
            max_sequence_length:  maximum length of sequence.
            num_token_predictions: number of LML
            mask_ratio: ratio of masking in LML
            buffer_size: how many data load to RAM.
            batch_size: batch size

        Return : tfdata

        """

        features = {
            "fasta": tf.io.FixedLenFeature([], tf.string),
            "seq": tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
        }
        if mode == "train":
            self.target_files = self.files[: self._train_len]
            train_files = self.target_files
            logging.info(
                f"you are using training dataset, list of training file : {train_files}"
            )
            dataset = tf.data.TFRecordDataset(
                train_files,
                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                compression_type="GZIP",
            )
        else:
            self.target_files = self.files[self._train_len :]
            valid_files = self.target_files
            logging.info(
                f"you are using validation dataset, list of training file : {valid_files}"
            )
            dataset = tf.data.TFRecordDataset(
                valid_files,
                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                compression_type="GZIP",
            )

        if is_training:
            dataset = dataset.shuffle(buffer_size, seed=self.seed)
            dataset = dataset.repeat()
        else:
            logging.info("you are using dataset for evaluation. no repeat, no shuffle")

        max_mask_num = int(num_token_predictions * 0.8)
        max_another_num = int(num_token_predictions * 0.1)

        def _parse_function(example_proto):
            with tf.device("cpu"):
                eg = tf.io.parse_single_example(example_proto, features)

                fasta = eg["fasta"]
                seq = tf.cast(eg["seq"], tf.int32)

                # length = eg['seq'].shape[0]
                length = len(
                    eg["seq"]
                )  # length means only number of AA, excluding the <cls> , <eos>

                # add cls, eos token
                seq = tf.concat(
                    [[aa_idx_vocab["<cls>"]], seq, [aa_idx_vocab["<eos>"]]], axis=0
                )
                mask = tf.ones(length + 2, dtype=tf.int32)

                # count
                num_mask = tf.cast(length, tf.float32) * mask_ratio * 0.8
                num_mask = tf.clip_by_value(
                    tf.cast(num_mask, tf.int32), 1, max_mask_num
                )

                num_replace = tf.cast(length, tf.float32) * mask_ratio * 0.1
                num_replace = tf.clip_by_value(
                    tf.cast(num_replace, tf.int32), 1, max_another_num
                )
                num_total = num_mask + 2 * num_replace

                # lm position.
                # range -> we have to exclude <cls>, <eos>
                lm_positions = tf.random.shuffle(tf.range(1, length + 1))[:num_total]
                masked_lm_positions = lm_positions[:num_total]

                # lm weight
                lm_weights = tf.ones(num_total, tf.int32)

                # new input
                masked_seq = tf.identity(seq)

                replacing_seq = tf.fill([num_mask], np.int32(aa_idx_vocab["<mask>"]))
                masked_seq = tf.tensor_scatter_nd_update(
                    masked_seq,
                    tf.expand_dims(masked_lm_positions[:num_mask], axis=-1),
                    replacing_seq,
                )

                replacing_seq = tf.random.uniform([num_replace], 0, 20, dtype=tf.int32)
                masked_seq = tf.tensor_scatter_nd_update(
                    masked_seq,
                    tf.expand_dims(
                        masked_lm_positions[num_mask : num_mask + num_replace], axis=-1
                    ),
                    replacing_seq,
                )
                # gt
                masked_lm_ids = tf.gather(seq, masked_lm_positions)

            return {
                "input_fasta": fasta,
                "input_seq": masked_seq,
                "input_seq_mask": mask,
                "input_lm_positions": masked_lm_positions,
                "input_lm_target": masked_lm_ids,
                "input_lm_weights": lm_weights,
                "length": length,
            }

        padded_shapes = {
            "input_fasta": [],
            "input_seq": [max_sequence_length],
            "input_seq_mask": [max_sequence_length],
            "input_lm_positions": [num_token_predictions],
            "input_lm_target": [num_token_predictions],
            "input_lm_weights": [num_token_predictions],
            "length": [],
        }
        zero = tf.constant(0, dtype=tf.int32)

        padded_value = {
            "input_fasta": "",
            "input_seq": tf.cast(aa_idx_vocab["<pad>"], tf.int32),
            "input_seq_mask": zero,
            "input_lm_positions": zero,
            "input_lm_target": np.int32(20),
            "input_lm_weights": zero,
            "length": zero,
        }

        dataset = dataset.map(
            _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=padded_shapes, padding_values=padded_value
        )
        dataset = dataset.prefetch(buffer_size)

        return dataset


# todo : uniref50 순서의 의미 -> 알파벳 순서인가?
