import os
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


import dataclasses
from official.core import config_definitions as cfg
from official.nlp.data import data_loader_factory
from official.nlp.data import data_loader
from official.core import input_reader
from alphafold.model.tf.proteins_dataset import parse_tfexample, create_tensor_dict

from pdp.data.afdata import FEATURES
from typing import Optional, Mapping, List, Sequence


from pdp.utils.vocab import aa_idx_vocab
from pdp.data.paths import get_expand_path

# @dataclasses.dataclass
# class BertPretrainDataConfig(cfg.DataConfig):
#   """Data config for BERT pretraining task (tasks/masked_lm)."""
#   input_path: str = ''
#   global_batch_size: int = 512
#   is_training: bool = True
#   seq_length: int = 512
#   max_predictions_per_seq: int = 76
#   use_next_sentence_label: bool = True
#   use_position_id: bool = False
#   # Historically, BERT implementations take `input_ids` and `segment_ids` as
#   # feature names. Inside the TF Model Garden implementation, the Keras model
#   # inputs are set as `input_word_ids` and `input_type_ids`. When
#   # v2_feature_names is True, the data loader assumes the tf.Examples use
#   # `input_word_ids` and `input_type_ids` as keys.
#   use_v2_feature_names: bool = False


@dataclasses.dataclass
class AFDataConfig(cfg.DataConfig):
    # Add fields here.
    input_path: str = get_expand_path("~/.cache/tfdata/pdb/*.tfrecord")
    global_batch_size: int = 16
    num_res: int = 256
    train_features: List[str] = dataclasses.field(
        default_factory=lambda: [
            "domain_name",
            "all_atom_positions",
            "all_atom_mask",
            "resolution",
            "old_aatype",
        ]
    )
    compression_type: str = "GZIP"
    is_training: bool = True
    bins: int = 16
    patch_size: int = 64
    vocab_size: int = len(aa_idx_vocab)

    epochs: int = 100
    step: int = 1000
    learning_rate: float = 1e-4
    end_learning_rate: float = 1e-6
    weight_decay_rate: float = 1e-5

    max_sequence_length: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_token_predictions: int = 128
    distance_hidden_size: int = 1024
    # drop_remainder
    # shuffle_buffer_size
    # deterministic
    # seed
    contact_k : int = 1


# example : https://github.com/tensorflow/models/blob/master/official/nlp/data/pretrain_dataloader.py#L48
from alphafold.model.tf.proteins_dataset import _make_features_metadata
from alphafold.model.tf.proteins_dataset import parse_reshape_logic

# re-implementation

# todo : write
@data_loader_factory.register_data_loader_cls(AFDataConfig)
class AFDataLoader(data_loader.DataLoader):
    def __init__(self, params):
        """Inits `BertPretrainDataLoader` class.
        Args:
          params: A `BertPretrainDataConfig` object.
        """
        self._params = params
        self._global_batch_size = params.global_batch_size
        self._num_res = params.num_res
        self._train_features = params.train_features
        self._features_metadata = _make_features_metadata(self._train_features)
        self._bins = params.bins
        self._contact_k = params.contact_k
        contact_mat = np.triu(
            np.ones([params.max_sequence_length, params.max_sequence_length]),
            k=self._contact_k,
        )
        self._contact_mat = contact_mat + contact_mat.T
        self._patch_size = params.patch_size
        self._max_sequence_length = params.max_sequence_length

    # from alphafold.model.tf.proteins_dataset import _make_features_metadata
    def _decode(self, record: tf.Tensor, use_TPU=False):
        # default features = ["aatype", "sequence", "seq_length"]
        feature_map = {
            k: tf.io.FixedLenSequenceFeature(shape=(), dtype=v[0], allow_missing=True)
            for k, v in self._features_metadata.items()
        }
        example = tf.io.parse_single_example(record, feature_map)
        print(example)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        if use_TPU:
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                example[name] = t
        return example

    def _parse(self, record: Mapping[str, tf.Tensor]):
        reshaped_features = parse_reshape_logic(record, self._features_metadata)
        transformed_feature = self._transform(reshaped_features)
        return transformed_feature

    #
    def _transform(self, parsed):

        fasta = parsed["domain_name"]
        # seq_string = parsed["sequence"]

        seq = parsed["old_aatype"]
        # todo : it is also matched another sequential residue [MACCCCXXXX] -> case 1. "CCC" case2. "XXX"
        idx1 = tnp.where(seq[:-1] == seq[1:])
        idx2 = tnp.where(seq[:-2] == seq[2:])

        # cut unknown region, it is corresponding to sequence length
        intersection = tf.sets.intersection(idx1, idx2)

        length = tf.reduce_min(intersection.values)
        length = tf.cast(length, tf.int32)
        length = tf.where(
            tf.size(intersection) == 0,
            len(seq),
            length,
        )

        # Sometimes, Residue is longer than max sequence length. So we will cut.
        length = tf.minimum(
            length, self._max_sequence_length - 2
        )  # we have to put <cls>, <eos> token

        # todo : pre-train과 똑같은 input을 만들기 위해서 cls, eos 추가.. 넣는게 맞는걸까 ?
        seq = tf.concat(
            [[aa_idx_vocab["<cls>"]], seq[:length], [aa_idx_vocab["<eos>"]]], axis=0
        )
        seq_mask = tf.ones(length + 2, dtype=tf.int64)

        # afdata doesn't include the secondary structure
        ss8 = tf.zeros(length + 2, dtype=tf.int64)
        ss_weight = tf.zeros(length + 2, dtype=tf.int64)

        # distance
        # we need only carbon alpha position (index 1)
        pos = parsed["all_atom_positions"][:length, 1]  # default dtype : tf.float32
        dist = tf.sqrt(tf.reduce_sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        nan_mask = tf.math.is_nan(dist) == [False]

        if self._contact_k > 0:
            diagonal_mask = tf.constant(self._contact_mat, tf.bool)[:length, :length]
            dist_mask = nan_mask & diagonal_mask
        else:
            dist_mask = nan_mask

        # todo : distance, nan 값 등장에 대한 분석
        dist = tf.math.multiply_no_nan(
            dist, tf.cast(dist_mask, tf.float32)
        )  # todo : nan 을 그냥 곱하면 안되는거 .. 정리하자.
        dist = tf.floor(dist)
        dist = tf.clip_by_value(dist - 2, 0, self._bins - 1)

        # zero padding to <cls> and <eos>
        dist_mask = tf.pad(dist_mask, [[1, 1], [1, 1]])

        dist = tf.pad(dist, [[1, 1], [1, 1]])
        # todo : think about "dtype"

        ij = tf.random.uniform(
            shape=[2],
            minval=1,
            maxval=tf.maximum(
                length + 2 - self._patch_size, 2
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
                dist[i : i + self._patch_size, j : j + self._patch_size], tf.int32
            ),
            "input_patch_dist_mask": tf.cast(
                dist_mask[i : i + self._patch_size, j : j + self._patch_size], tf.int32
            ),
            "input_ij": tf.cast(ij, tf.int32),
            "input_length": tf.reshape(length, [1]),
        }

    def _batch_fn(
        self,
        dataset: tf.data.Dataset,
        input_context: tf.distribute.InputContext = None,
    ):
        padded_shapes = {
            "input_fasta": [1],
            "input_seq": [self._max_sequence_length],
            "input_seq_mask": [self._max_sequence_length],
            "input_ss8_target": [self._max_sequence_length],
            "input_ss_weight": [self._max_sequence_length],
            "input_dist_target": [self._max_sequence_length, self._max_sequence_length],
            "input_dist_mask": [self._max_sequence_length, self._max_sequence_length],
            "input_ij": [2],
            "input_length": [1],
            "input_patch_dist_target": [self._patch_size, self._patch_size],
            "input_patch_dist_mask": [self._patch_size, self._patch_size],
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

        dataset = dataset.padded_batch(
            self._global_batch_size,
            padded_shapes=padded_shapes,
            padding_values=padded_value,
        )
        return dataset

    def load(
        self, input_context: Optional[tf.distribute.InputContext] = None
    ) -> tf.data.Dataset:
        """Returns a tf.dataset.Dataset."""
        # decode -> parse -> transform -> batch
        # todo : write the transform function
        # ref 1. https://github.com/deepmind/alphafold/blob/0be2b30b98f0da7aecb973bde04758fae67eb913/alphafold/model/tf/input_pipeline.py#L33
        # ref 2. https://github.com/tensorflow/models/blob/master/official/nlp/data/pretrain_dataloader.py#L227

        # todo : I have to know detail of input_context.
        reader = input_reader.InputReader(
            params=self._params,
            # decoder_fn=lambda x: create_tensor_dict(x, FEATURES),
            decoder_fn=self._decode,
            parser_fn=self._parse,
            dataset_fn=lambda x: tf.data.TFRecordDataset(
                x, compression_type=self._params.compression_type
            ),
            transform_and_batch_fn=self._batch_fn,
        )  # transform_and_batch_fn=self._transform_and_batch_fn

        return reader.read(input_context)


# my_data_config = AFDataConfig()
# my_loader = data_loader_factory.get_data_loader(my_data_config)
# my_loader.load()
# my_data_config = MyDataConfig()
# # Returns MyDataLoader(my_data_config).
#
# my_loader = data_loader_factory.get_data_loader(my_data_config)
# my_loader.load()
