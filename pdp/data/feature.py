from typing import Sequence, Tuple, List, Text


from Bio.PDB.vectors import Vector
from pdp.utils import vocab
import numpy as np

# todo : input 에서 gap(-), X, .. 등 모르는 것들 어떻게 처리할건지.
# todo : X 는 unknown 인 것 같았는데 weight 는 어떻게 줄건지.


# todo : 3-letter 추가하기
class AminoAcid(str):
    # todo : 아미노산의 정의. 표현식 3 a. 1-letter  b. 3-letter  c. index
    """
    desc :
    """

    def __init__(self, aa):

        if isinstance(aa, str):
            self.sequence = aa
            # todo : 아미노산은 사실 string 이면 되는데 .. 데이터 편의를 위해서 index를 추가 해놓는게 맞을까? 별로 크지 않아서 상관없지만 .. 그래도 낭비는 낭비일지도 ?
            self._idx = self._aa_idx(aa)

        elif isinstance(aa, np.ndarray):
            self.sequence = self._idx_aa(aa)
            self._idx = aa

        else:
            raise ValueError(f"{aa} is not supported type.")

    def __str__(self):
        return self.sequence

    def get_idx(self):
        return self._idx

    # todo : init에 포함 시킬지 말지
    def _vocab_check(self, sequence) -> None:
        """
        check the key error.
        """
        for i, current_seq in enumerate(sequence):
            if current_seq not in vocab.aa_idx_vocab:
                raise ValueError(
                    f'"{current_seq}" is not existed in vocab. idx number : {i}'
                )

    def _aa_idx(self, sequence: str) -> List[int]:
        """
        convert the amino acid to index.
        If it has unknown amino acid, it is replaced to <unk> token.
        """

        index_list = []
        unknown_token = vocab.aa_idx_vocab["<unk>"]

        for current_seq in sequence:
            # todo : unknwon 토큰일때 무언가 기록하거나.. 해야 하지 않을까 ?
            current_idx = vocab.aa_idx_vocab.get(current_seq, unknown_token)
            index_list.append(current_idx)

        # todo : 어떻게 짠게 더 보기 편할까 ?
        # Equal to
        # [vocab.aa_idx_vocab.get(_seq, vocab.aa_idx_vocab['<unk>']) for _seq in self]
        return index_list

    def _idx_aa(self, idx: List[int]) -> str:
        """
        convert the index to amino acid.
        """
        seq_list = []
        for current_idx in idx:
            current_seq = vocab.idx_aa_vocab.get(current_idx)
            seq_list.append(current_seq)

        nonlist_seq = "".join(seq_list)

        # Equal to
        # "".join([idx_aa_vocab[_idx] for _idx in idx])
        return nonlist_seq

    def __repr__(self):
        return self.sequence

    def __eq__(self, other):
        return self.sequence == other


class SS3(object):
    # todo : 정의, 표현식 a. letter b. index , fullname = 1-letter 헷갈릴 여지 없으므로 적지 않는다. SS8 관계
    """ """

    def __init__(self, ss3, vocab_check: bool = False):
        if isinstance(ss3, str):
            self.ss3 = ss3
            self._idx = self._ss3_idx(ss3)

        elif isinstance(ss3, list):
            self.ss3 = self._idx_ss3(ss3)
            self._idx = ss3

        else:
            raise ValueError(f"{ss3} is not supported type.")

        if vocab_check:
            self.vocab_check()

    def get_idx(self):
        """
        convert the 3-class secondary structure to index.
        If it has unknown secondary structure, it is replaced to <unk> token.
        >>> ss3 = "HHHCCC"
        >>> ss3_idx(ss3)
        >>> [0, 0, 0, 2, 2, 2]
        :return:
        """
        return self._idx

    def vocab_check(self) -> None:
        for i, current_ss3 in enumerate(self.ss3):
            if current_ss3 not in vocab.ss3_idx_vocab:
                raise ValueError(
                    f'"{current_ss3}" is not existed in vocab. idx number : {i}'
                )

    def _ss3_idx(self, ss3: str) -> List[int]:
        """
        convert the 3-class secondary structure to index.
        If it has unknown secondary structure, it is replaced to <unk> token.
        """

        return [vocab.ss3_idx_vocab[_ss3] for _ss3 in ss3]

    # todo : self를 딱히 쓰지 않아도 이렇게 쓰는게 나을까 ? class function 이 나을까 class function은 외부에서 사용하고 싶을때 쓰는거 아닌가
    def _idx_ss3(self, idx: List[int]) -> List[str]:
        """
        convert the 3ss index to 3-class secondary structure
        >>> idx = [0, 0, 0, 2, 2, 2]
        >>> idx_ss3(idx)
        >>> "HHHCCC"
        """
        return "".join([vocab.idx_ss3_vocab[_idx] for _idx in idx])

    def __repr__(self):
        return self.ss3

    def __eq__(self, other):
        return self.ss3 == other


class SS8(object):
    # todo : 정의, 표현식 a. letter b. index , fullname = 1-letter 헷갈릴 여지 없으므로 적지 않는다. SS3 관계
    def __init__(self, ss8, vocab_check: bool = False):

        if isinstance(ss8, str):
            self.ss8 = ss8
            self._idx = self._ss8_idx(ss8)

        elif isinstance(ss8, tuple):
            self.ss8 = self._idx_ss8(ss8)
            self._idx = ss8

        else:
            raise ValueError(f"{ss8} is not supported type.")

        if vocab_check:
            self.vocab_check()

    def get_ss3(self) -> SS3:
        """
        convert the 8-class secondary structure to 3-class secondary structure.
        """
        return SS3("".join([vocab.ss8_ss3_vocab[_seq] for _seq in self.ss8]))

    def get_idx(self):
        """
        convert the 8-class secondary structure to index.
        If it has unknown secondary structure, it is replaced to <unk> token.
        """
        return self._idx

    # todo : raise 문은 예외처리로 동작가능하게끔 .. assert는 내부 정확성을 위해서?  https://google.github.io/styleguide/pyguide.html#244-decision 읽어보자.

    def vocab_check(self) -> None:
        for i, current_ss8 in enumerate(self.ss8):
            if current_ss8 not in vocab.ss8_idx_vocab:
                raise ValueError(
                    f'"{current_ss8}" is not existed in vocab. idx number : {i}'
                )

    # ss8
    def _ss8_idx(self, ss8: str) -> List[int]:
        """
        convert the 8-class secondary structure to index.
        If it has unknown secondary structure, it is replaced to <unk> token.
        """
        return [vocab.ss8_idx_vocab[_ss8] for _ss8 in ss8]

    def _idx_ss8(self, idx: List[int]) -> List[str]:
        """
        convert the ss8 index to 8-class secondary structure
        """
        return "".join([vocab.idx_ss8_vocab[_idx] for _idx in idx])

    def __repr__(self):
        return self.ss8

    def __eq__(self, other):
        return self.ss8 == other
