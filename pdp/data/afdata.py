import io
import dataclasses
import logging
from typing import Any, Mapping, Optional

# from alphafold.common.protein import Protein as alphafold_protein
from Bio.PDB import PDBParser
from alphafold.common.protein import PDB_MAX_CHAINS
from alphafold.model.tf import protein_features as pf
from alphafold.common import residue_constants
from alphafold.model import all_atom
import numpy as np
import tensorflow as tf

from pdp.data.utils import _bytes_feature, _int64_feature, _float_feature
from pdp.data.helper import AbstractDataclass
import copy

SHRUNK_FEATURE = [
    "aatype",
    "domain_name",
    "seq_length",
    "sequence",
    "all_atom_positions",
    "all_atom_mask",
    "resolution",
]
FEATURES = {_feature: pf.FEATURES[_feature] for _feature in SHRUNK_FEATURE}
FEATURES.update({"old_aatype": (tf.int64, [pf.NUM_RES])})
pf.FEATURES = FEATURES
FEATURE_TYPES = {k: v[0] for k, v in FEATURES.items()}
FEATURE_SIZES = {k: v[1] for k, v in FEATURES.items()}


def _to_feature(array: np.array):
    """
    convert array to tf.train.Feature
    """
    # todo : write tfdata tutorial, flatten() vs 2d array -> serialize -> string
    # Feature allow to only 1-d array, if it has more than 2-dimension, it must to be flatten.
    # So when you read the data, data must be reverted to original shape.
    # check the "parse_reshape_logic"
    if len(array.shape) > 1:
        array = array.flatten()

    if array.dtype == np.int64:
        tensor = _int64_feature(array)

    elif array.dtype == np.float64:
        tensor = _float_feature(array)

    # In numpy, "U" and "S" type are string.
    elif array.dtype.kind in ["U", "S"]:
        tensor = _bytes_feature([array[0].encode()])

    else:
        raise Exception(f"Input Array Data type : {array.dtype}. It is not supported.")

    return tensor


@dataclasses.dataclass(frozen=True)
class AFData(AbstractDataclass):
    """Protein structure representation."""

    # todo : think about another feature.

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    all_atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    all_atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # todo : set description.
    resolution: np.ndarray  # [1]

    # one-letter Amino acid.
    # "X" equal to 20 in aatype
    sequence: np.ndarray  # [num_res]

    # todo : does domain_name means that pdb name with chain index ?
    domain_name: np.ndarray  # [1]

    seq_length: np.ndarray  # [num_res]

    old_aatype: np.ndarray  # [num_res]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )

    # todo : is this functions need?,
    # If transforming is essentially equal but just different value,
    # i think that it is comfortable that include this function into class.
    def atom37_to_torsion_angles(self):
        torsions = all_atom.atom37_to_torsion_angles(
            self.aatype[None, :], self.atom_positions[None, :], self.atom_mask[None, :]
        )
        return torsions

    def atom37_to_frames(self):
        frames = all_atom.atom37_to_frames(
            self.aatype[None, :], self.atom_positions[None, :], self.atom_mask[None, :]
        )
        return frames

    def get_example(self) -> tf.train.Example:
        # todo : multiple chain.
        feature = {k: _to_feature(v) for k, v in self.__dict__.items()}
        feature["aatype"] = np_one_hot(
            self.aatype.astype(np.int), 21
        )  # include 'X' (index number = 20)
        feature["aatype"] = _to_feature(feature["aatype"].flatten())

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto


def np_one_hot(indices, depth):
    one_hot = np.eye(depth)[indices]
    return one_hot


from pdp.utils import vocab


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> AFData:
    # https://github.com/deepmind/alphafold/blob/0be2b30b98f0da7aecb973bde04758fae67eb913/alphafold/common/protein.py
    # Copyright 2021 DeepMind Technologies Limited
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    sequence = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    old_aatype = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        temp_seq = []
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            # todo : vocabulary
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            old_seq_idx = vocab.aa_idx_vocab.get(
                res_shortname, vocab.aa_idx_vocab["<unk>"]
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            temp_seq.append(res_shortname)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
            old_aatype.append(old_seq_idx)

        sequence.append("".join(temp_seq))

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    domain_name = np.array(
        [
            parser.get_header()["idcode"] + "_" + chain_id
            for chain_id in unique_chain_ids
        ],
        dtype=np.str,
    )
    sequence = np.array(sequence)
    # resolution is sometime empty.
    if parser.get_header()["resolution"] is None:
        resolution = np.array([999.0] * len(unique_chain_ids))
    else:
        resolution = np.array(
            [parser.get_header()["resolution"]] * len(unique_chain_ids)
        )
    return AFData(
        all_atom_positions=np.array(atom_positions),
        all_atom_mask=np.array(atom_mask, dtype=np.int),
        aatype=np.array(aatype, dtype=np.float),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        resolution=resolution,
        sequence=sequence,
        domain_name=domain_name,
        seq_length=np.array([len(seq) * [len(seq)] for seq in sequence]),
        old_aatype=np.array(old_aatype),
    )
