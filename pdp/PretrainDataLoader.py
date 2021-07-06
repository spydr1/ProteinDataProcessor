import tensorflow as tf
import logging
from pdp.vocab import aa2idx_vocab
import numpy as np

# new span
#
class PretrainDataLoader:
    def __init__(self,
                 file,
                 is_training=False,
                 max_sequence_length=512,
                 max_mlm_length=100,
                 mask_ratio=0.2,
                 buffer_size=200,
                 batch_size=8
                 ):
        # todo : numpy, tfrecord
        pass

    def download(self):
        pass

    # todo : how to set span option.
    # todo : tfx
    def load(self, span : int) ->tf.data.TFRecordDataset :
        pass
        # 1. pre-processing, parsing.

def load_tfrecord(file,
                  is_training=False,
                  max_sequence_length= 512,
                  max_mlm_length=100,
                  mask_ratio = 0.2,
                  buffer_size=200,
                  batch_size=8):

    features = {
        'fasta': tf.io.FixedLenFeature([], tf.string),
        'seq': tf.io.RaggedFeature(value_key="seq", dtype=tf.int64),
    }
    dataset = tf.data.TFRecordDataset(file,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                      compression_type="GZIP",)

    if is_training:
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
    else:
        logging.info("you are using dataset for evaluation. no repeat, no shuffle")

    def _parse_function(example_proto):
        with tf.device('cpu'):
            eg = tf.io.parse_single_example(example_proto, features)

            fasta = eg['fasta']
            seq = eg['seq']
            length = len(eg['seq']) # length means only number of AA, excluding the <cls> , <eos>

            # add cls, eos token
            seq = tf.concat([[aa2idx_vocab['<cls>']], seq, [aa2idx_vocab['<eos>']]], axis=0)
            mask = tf.ones(length + 2, dtype=tf.int64)

            # count
            num_lm = tf.minimum(int(tf.cast(length, tf.float32) * mask_ratio), max_mlm_length)

            # lm position.
            # range -> we have to exclude <cls>, <eos>
            lm_positions = tf.random.shuffle(tf.range(1, length + 1))[:num_lm]
            masked_lm_positions = lm_positions[:num_lm]

            # lm weight
            lm_weights = tf.ones(num_lm, tf.int64)

            # new input
            masked_seq = tf.identity(seq)

            masked_seq = tf.fill(tf.shape(lm_weights), np.int64(aa2idx_vocab['<mask>']))
            masked_seq = tf.tensor_scatter_nd_update(masked_seq,
                                                     tf.expand_dims(masked_lm_positions, axis=-1),
                                                     masked_seq)
            # gt
            masked_lm_ids = tf.gather(seq, lm_positions)

        return {"input_fasta": fasta,
                 "input_seq":masked_seq,
                 "input_mask":mask,
                 "input_lm_positions":lm_positions,
                 "input_target":masked_lm_ids,
                 "input_lm_weights": lm_weights}

    padded_shapes = {
        "input_fasta" : [],
        "input_seq" : [max_sequence_length],
        "input_mask" : [max_sequence_length],
        "input_lm_positions": [max_mlm_length],
        "input_target": [max_mlm_length],
        "input_lm_weights": [max_mlm_length]
    }

    dataset = dataset.map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# todo : pretrain(MLM)
# todo : full data : coords, angs , ss
# todo : load train, eval set {1,2,3,4,5}

# data -> object -> tfrecord -> tfx