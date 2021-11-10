import os

import tensorflow as tf
import logging
from pdp.utils.vocab import aa_idx_vocab
import numpy as np

# new span
#
class FullDataLoader:
    def __init__(self,
                 is_training=False,
                 max_sequence_length=512,
                 max_mlm_length=100,
                 mask_ratio=0.15,
                 buffer_size=200,
                 batch_size=8,
                 bins=16,
                 split_level='superfamily',
                 cv_partition='4',
                 split='train',
                 ):
        # todo : numpy, tfrecord
        self.is_training=is_training
        self.max_sequence_length=max_sequence_length
        self.max_mlm_length=max_mlm_length
        self.mask_ratio=mask_ratio
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.split_level = split_level
        self.cv_partition = cv_partition
        self.split = split
        self.bins= bins


    def _check_exist(self):
        tfrecord_file = f"~/.cache/tfdata/"
        return os.path.exists(tfrecord_file)

    def _download(self,
                 ):
        import urllib.request
        url = "https://drive.google.com/drive/folders/12hwbZwdwUYNaenUqJL7ybyHqsbNP0_C_?usp=sharing"
        urllib.request.urlretrieve(url, "~/.cache/tfdata/")

    # todo : how to set span option.
    # todo : tfx
    # todo : think about return
    def load_tfrecord(self,
                      seed=12345) -> tf.data.TFRecordDataset :
        if self._check_exist():
            pass
        else :
            self._download()

        if self.is_training :
            _partition_list = ['0', '1', '2', '3', '4']
            tfrecord_files = []
            desc = ""
            for _partition in _partition_list:

                if self.cv_partition != _partition:
                    desc+=f"{_partition} "
                    tfrecord_files.append(f'~/.cache/tfdata/{self.split_level}/{_partition}.tfrecord')
                logging.info(f"partiion list : {desc}")

        else :
            tfrecord_files = f'~/.cache/tfdata/{self.split_level}/{self.cv_partition}.tfrecord'

        features = {
            'seq': tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
            'dist': tf.io.FixedLenFeature([], tf.string),
            'coords': tf.io.FixedLenFeature([], tf.string),
            'ss3': tf.io.RaggedFeature(value_key="ss3", dtype=tf.int64),
            'ss8': tf.io.RaggedFeature(value_key="ss8", dtype=tf.int64),
            'ss_weights': tf.io.RaggedFeature(value_key="ss_weights", dtype=tf.int64)
        }
        dataset = tf.data.TFRecordDataset(tfrecord_files,
                                          num_parallel_reads=tf.data.experimental.AUTOTUNE,)
                                          # compression_type="GZIP", )

        if self.is_training:
            dataset = dataset.shuffle(self.buffer_size, seed=seed)
            dataset = dataset.repeat()
        else:
            logging.info("you are using dataset for evaluation. no repeat, no shuffle")

        triangle_mat = np.triu(np.ones([self.max_sequence_length, self.max_sequence_length]), k=1)
        triangle_mat[0, :] = 0
        triangle_mat[:, 0] = 0

        max_mask_num = int(self.max_mlm_length * 0.8)
        max_another_num = int(self.max_mlm_length * 0.1)

        def _parse_function(example_proto):
            with tf.device('cpu'):
                eg = tf.io.parse_single_example(example_proto, features)

                seq = tf.cast(eg['seq'], tf.int32)

                ss3 = tf.cast(eg['ss3'], tf.int32)
                ss3 = tf.concat([[0], ss3], axis=0)

                ss8 = tf.cast(eg['ss8'], tf.int32)
                ss8 = tf.concat([[0], ss8], axis=0)

                length = len(eg['seq'])  # length means only number of AA, excluding the <cls> , <eos>

                # add cls, eos token
                seq = tf.concat([[aa_idx_vocab['<cls>']], seq, [aa_idx_vocab['<eos>']]], axis=0)
                mask = tf.ones(length + 2, dtype=tf.int32)

                ss_weights = tf.cast(eg['ss_weights'], tf.int32)
                # set the weight of cls token.
                ss_weights = tf.concat([[0], ss_weights], axis=0)

                # distance
                dist = tf.io.parse_tensor(eg['dist'], out_type=tf.float32)
                nan_mask = tf.cast(tf.math.is_nan(dist) == False, tf.int32)
                dist = tf.cast(dist, dtype=tf.int32) * nan_mask

                # distance range is 2~18
                # clipping range is 0~16
                dist = tf.clip_by_value(dist, 2, self.bins + 1) - 2

                # distance mask
                dist_mask = tf.constant(triangle_mat, tf.int32)[:length, :length] * nan_mask

                # "cls" pad
                dist = tf.pad(dist, [[1, 0], [1, 0]])
                dist_mask = tf.pad(dist_mask, [[1, 0], [1, 0]])

                num_mask = tf.cast(length, tf.float32) * self.mask_ratio * 0.8
                num_mask = tf.clip_by_value(tf.cast(num_mask, tf.int32), 1, max_mask_num)

                num_replace = tf.cast(length, tf.float32) * self.mask_ratio * 0.1
                num_replace = tf.clip_by_value(tf.cast(num_replace, tf.int32), 1, max_another_num)
                num_total = num_mask + 2 * num_replace

                # lm position.
                # range -> we have to exclude <cls>, <eos>
                lm_positions = tf.random.shuffle(tf.range(1, length + 1))[:num_total]
                masked_lm_positions = lm_positions[:num_total]

                # lm weight
                lm_weights = tf.ones(num_total, tf.int32)

                # new input
                masked_seq = tf.identity(seq)

                replacing_seq = tf.fill([num_mask], np.int32(aa_idx_vocab['<mask>']))
                masked_seq = tf.tensor_scatter_nd_update(masked_seq,
                                                         tf.expand_dims(masked_lm_positions[:num_mask], axis=-1),
                                                         replacing_seq)

                replacing_seq = tf.random.uniform([num_replace], 0, 20, dtype=tf.int32)
                masked_seq = tf.tensor_scatter_nd_update(masked_seq,
                                                         tf.expand_dims(
                                                             masked_lm_positions[num_mask:num_mask + num_replace],
                                                             axis=-1),
                                                         replacing_seq)
                # gt
                masked_lm_ids = tf.gather(seq, lm_positions)

            return {"input_seq": masked_seq,
                    "input_seq_mask": mask,  # exclude "cls", "eos"

                    "input_lm_positions": lm_positions,
                    "input_lm_target": masked_lm_ids,
                    "input_lm_weights": lm_weights,

                    "input_ss3_target": ss3,
                    "input_ss8_target": ss8,
                    "input_ss_weights": ss_weights,  # exclude "cls", "eos"

                    "input_dist_target": dist,
                    "input_dist_mask": dist_mask}  # exclude "cls", "eos"

        padded_shapes = {
            "input_seq": [self.max_sequence_length],
            "input_seq_mask": [self.max_sequence_length],
            "input_lm_positions": [self.max_mlm_length],
            "input_lm_target": [self.max_mlm_length],
            "input_lm_weights": [self.max_mlm_length],
            "input_ss3_target": [self.max_sequence_length],
            "input_ss8_target": [self.max_sequence_length],
            "input_ss_weights": [self.max_sequence_length],
            "input_dist_target": [self.max_sequence_length, self.max_sequence_length],
            "input_dist_mask": [self.max_sequence_length, self.max_sequence_length]
        }

        zero = tf.constant(0, dtype=tf.int32)

        padded_value = {
            "input_seq": tf.cast(aa_idx_vocab['<pad>'], tf.int32),
            "input_seq_mask": zero,
            "input_lm_positions": zero,
            "input_lm_target": zero,
            "input_lm_weights": zero,
            "input_ss3_target": zero,
            "input_ss8_target": zero,
            "input_ss_weights": zero,
            "input_dist_target": zero,
            "input_dist_mask": zero
        }

        dataset = dataset.map(
            _parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padded_value)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset



        # 1. pre-processing, parsing.

def load_tfrecord(file,
                  is_training=False,
                  max_sequence_length= 512,
                  num_token_predictions=100,
                  mask_ratio = 0.2,
                  buffer_size=200,
                  batch_size=8,
                  bins = 16,
                  seed=12345):
    features = {
        'seq': tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
        'dist': tf.io.FixedLenFeature([], tf.string),
        'coords': tf.io.FixedLenFeature([], tf.string),
        'ss3': tf.io.RaggedFeature(value_key="ss3", dtype=tf.int64),
        'ss8': tf.io.RaggedFeature(value_key="ss8", dtype=tf.int64),
        'ss_weights': tf.io.RaggedFeature(value_key="ss_weights", dtype=tf.int64)
    }
    dataset = tf.data.TFRecordDataset(file,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                      compression_type="GZIP",)

    if is_training:
        dataset = dataset.shuffle(buffer_size, seed=seed)
        dataset = dataset.repeat()
    else:
        logging.info("you are using dataset for evaluation. no repeat, no shuffle")

    triangle_mat = np.triu(np.ones([max_sequence_length, max_sequence_length]), k=1)
    triangle_mat[0,:] = 0
    triangle_mat[:,0] = 0

    max_mask_num = int(num_token_predictions * 0.8)
    max_another_num = int(num_token_predictions * 0.1)

    def _parse_function(example_proto):
        with tf.device('cpu'):
            eg = tf.io.parse_single_example(example_proto, features)

            seq = tf.cast(eg['seq'], tf.int32)

            ss3 = tf.cast(eg['ss3'],tf.int32)
            ss3 = tf.concat([[0], ss3], axis=0)

            ss8 = tf.cast(eg['ss8'],tf.int32)
            ss8 = tf.concat([[0], ss8], axis=0)

            length = len(eg['seq'])  # length means only number of AA, excluding the <cls> , <eos>

            # add cls, eos token
            seq = tf.concat([[aa_idx_vocab['<cls>']], seq, [aa_idx_vocab['<eos>']]], axis=0)
            mask = tf.ones(length + 2, dtype=tf.int32)

            ss_weights = tf.cast(eg['ss_weights'],tf.int32)
            # set the weight of cls token.
            ss_weights = tf.concat([[0], ss_weights], axis=0)

            # distance
            dist = tf.io.parse_tensor(eg['dist'], out_type=tf.float32)
            nan_mask = tf.cast(tf.math.is_nan(dist) == False, tf.int32)
            dist = tf.cast(dist, dtype=tf.int32) * nan_mask

            # distance range is 2~18
            # clipping range is 0~16
            dist = tf.clip_by_value(dist, 2, bins + 1) - 2

            # distance mask
            dist_mask = tf.constant(triangle_mat, tf.int32)[:length, :length] * nan_mask
            # "cls" pad
            dist = tf.pad(dist, [[1, 0], [1, 0]])
            dist_mask = tf.pad(dist_mask, [[1, 0], [1, 0]])

            num_mask = tf.cast(length, tf.float32) * mask_ratio * 0.8
            num_mask = tf.clip_by_value(tf.cast(num_mask, tf.int32), 1, max_mask_num)

            num_replace = tf.cast(length, tf.float32) * mask_ratio * 0.1
            num_replace = tf.clip_by_value(tf.cast(num_replace, tf.int32), 1, max_another_num)
            num_total = num_mask + 2 * num_replace

            # lm position.
            # range -> we have to exclude <cls>, <eos>
            lm_positions = tf.random.shuffle(tf.range(1, length + 1))[:num_total]
            masked_lm_positions = lm_positions[:num_total]

            # lm weight
            lm_weights = tf.ones(num_total, tf.int32)

            # new input
            masked_seq = tf.identity(seq)

            replacing_seq = tf.fill([num_mask], np.int32(aa_idx_vocab['<mask>']))
            masked_seq = tf.tensor_scatter_nd_update(masked_seq,
                                                     tf.expand_dims(masked_lm_positions[:num_mask], axis=-1),
                                                     replacing_seq)

            replacing_seq = tf.random.uniform([num_replace], 0, 20, dtype=tf.int32)
            masked_seq = tf.tensor_scatter_nd_update(masked_seq,
                                                     tf.expand_dims(
                                                         masked_lm_positions[num_mask:num_mask + num_replace], axis=-1),
                                                     replacing_seq)
            # gt
            masked_lm_ids = tf.gather(seq, lm_positions)

        return {"input_seq":masked_seq,
                "input_seq_mask":mask, # exclude "cls", "eos"

                "input_lm_positions":lm_positions,
                "input_lm_target":masked_lm_ids,
                "input_lm_weights": lm_weights,

                "input_ss3_target":ss3,
                "input_ss8_target":ss8,
                "input_ss_weights":ss_weights, # exclude "cls", "eos"

                "input_dist_target":dist,
                "input_dist_mask":dist_mask} # exclude "cls", "eos"

    padded_shapes = {
        "input_seq" : [max_sequence_length],
        "input_seq_mask" : [max_sequence_length],
        "input_lm_positions": [num_token_predictions],
        "input_lm_target": [num_token_predictions],
        "input_lm_weights": [num_token_predictions],
        "input_ss3_target": [max_sequence_length],
        "input_ss8_target": [max_sequence_length],
        "input_ss_weights": [max_sequence_length],
        "input_dist_target": [max_sequence_length,max_sequence_length],
        "input_dist_mask": [max_sequence_length, max_sequence_length]
    }

    zero = tf.constant(0, dtype=tf.int32)

    padded_value = {
        "input_seq": tf.cast(aa_idx_vocab['<pad>'], tf.int32),
        "input_seq_mask": zero,
        "input_lm_positions": zero,
        "input_lm_target": zero,
        "input_lm_weights": zero,
        "input_ss3_target": zero,
        "input_ss8_target": zero,
        "input_ss_weights": zero,
        "input_dist_target": zero,
        "input_dist_mask": zero
    }


    dataset = dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padded_value)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
