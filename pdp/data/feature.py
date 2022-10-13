from typing import List
from pdp.utils import vocab
import numpy as np

# todo : input 에서 gap(-), X, .. 등 모르는 것들 어떻게 처리할건지.
# todo : X 는 unknown 인 것 같았는데 weight 는 어떻게 줄건지.


# todo : 3-letter 추가하기
class AminoAcid(str):
    # todo : 아미노산의 정의. 표현식 3 a. 1-letter  b. 3-letter  c. index
    """
    Amino acid data

    Arga:
        sequence: 1-letter amino acid.
        idx: one-hot vector of amino acid.
    """

    def __init__(self, aa):
        if isinstance(aa, str):
            self.aa = aa
            # todo : aminoacid 클래스 안에 aa, attribute ... 무언가 별로인데 ?
            # todo : 아미노산은 사실 string 이면 되는데 .. 데이터 편의를 위해서 index를 추가 해놓는게 맞을까? 별로 크지 않아서 상관없지만 .. 그래도 낭비는 낭비일지도 ?
            self.idx = self._aa_idx()

        elif isinstance(aa, np.ndarray):
            self.idx = aa.tolist()
            self.aa = self._idx_aa()

        elif isinstance(aa, list):
            self.idx = aa
            self.aa = self._idx_aa()

        else:
            raise ValueError(
                f"{aa} is not supported type. It must be string or list format."
            )

    def __str__(self):
        return self.aa

    # todo : init에 포함 시킬지 말지
    def _vocab_check(self, sequence) -> None:
        """
        check the key error.
        """
        vocabulary = vocab.aa_idx_vocab

        for i, current_seq in enumerate(sequence):
            if current_seq not in vocabulary:
                raise ValueError(
                    f'"{current_seq}" is not existed in vocab. idx number : {i}'
                )

    def _aa_idx(self) -> List[int]:
        """
        convert the amino acid to index.
        If it has unknown amino acid, it is replaced to <unk> token.
        """

        index_list = []
        unknown_token = vocab.aa_idx_vocab["X"]
        vocabulary = vocab.aa_idx_vocab

        for seq in self.aa:
            # todo : unknown 토큰일때 무언가 기록하거나.. 해야 하지 않을까 ?
            current_idx = vocabulary.get(seq, unknown_token)
            index_list.append(current_idx)

        # todo : 어떻게 짠게 더 보기 편할까 ?
        # Equal to
        # [vocab.aa_idx_vocab.get(seq, vocab.aa_idx_vocab['<unk>']) for seq in self.aa]
        return index_list

    def _idx_aa(self) -> str:
        """
        convert the index to amino acid.
        """
        seq_list = []
        vocabulary = vocab.idx_aa_vocab

        for idx in self.idx:
            current_seq = vocabulary.get(idx)
            seq_list.append(current_seq)

        seq = "".join(seq_list)

        # Equal to
        # "".join([idx_aa_vocab[_idx] for _idx in idx])
        return seq

    def __repr__(self):
        return self.aa

    def __eq__(self, other):
        return self.aa == other

    def get_3_letter(self):
        vocabulary = vocab.aa_dict_inv
        return list([vocabulary[aa] for aa in self.aa])


class SS3(object):
    # todo : 정의, 표현식 a. letter b. index , fullname = 1-letter 헷갈릴 여지 없으므로 적지 않는다. SS8 관계
    """
    3-class secondary structure.

    Args:
        ss3: 3-class secondary structure. If it has unknown secondary structure, it is replaced to <X> token.
        idx: one-hot vector of ss3.
    """

    def __init__(self, ss3, vocab_check: bool = False):

        if isinstance(ss3, str):
            self.ss3 = ss3
            self.idx = self._ss3_idx()
        elif isinstance(ss3, np.ndarray) or isinstance(ss3, list):
            self.idx = ss3
            self.ss3 = self._idx_ss3()
        else:
            raise ValueError(f"{ss3} is not supported type.")

        if vocab_check:
            self.vocab_check()

    def vocab_check(self) -> None:
        """
        check the key error.
        """

        vocabulary = vocab.ss3_idx_vocab

        for i, current_ss3 in enumerate(self.ss3):
            if current_ss3 not in vocabulary:
                raise ValueError(
                    f'"{current_ss3}" is not existed in vocab. idx number : {i}'
                )

    def _ss3_idx(self) -> List[int]:
        """
        convert the 3-class secondary structure to index.
        If it has unknown secondary structure, it is replaced to <unk> token.
        """

        vocabulary = vocab.ss3_idx_vocab

        return [vocabulary[ss3] for ss3 in self.ss3]

    # todo : self를 딱히 쓰지 않아도 이렇게 쓰는게 나을까 ? class function 이 나을까 class function은 외부에서 사용하고 싶을때 쓰는거 아닌가
    def _idx_ss3(self) -> List[str]:
        """
        convert the 3ss index to 3-class secondary structure
        """
        vocabulary = vocab.idx_ss3_vocab
        return "".join([vocabulary[idx] for idx in self.idx])

    def __repr__(self):
        return self.ss3

    def __eq__(self, other):
        return self.ss3 == other


class SS8(object):
    """
    8-class secondary structure.

    Args:
        ss8: 8-class secondary structure. If it has unknown secondary structure, it is replaced to <X> token.
        idx: one-hot vector of ss8.
    """

    # todo : 정의, 표현식 a. letter b. index , fullname = 1-letter 헷갈릴 여지 없으므로 적지 않는다. SS3 관계
    def __init__(self, ss8, vocab_check: bool = False):

        if isinstance(ss8, str):
            self.ss8 = ss8
            self.idx = self._ss8_idx(ss8)
        elif isinstance(ss8, np.ndarray) or isinstance(ss8, list):
            self.ss8 = self._idx_ss8(ss8)
            self.idx = ss8
        else:
            raise ValueError(f"{ss8} is not supported type.")

        if vocab_check:
            self.vocab_check()

    def get_ss3(self) -> SS3:
        """
        convert the 8-class secondary structure to 3-class secondary structure.
        """
        vocabulary = vocab.ss8_ss3_vocab
        unknown_token = vocabulary["X"]

        seq_list = [vocabulary.get(_seq, unknown_token) for _seq in self.ss8]

        return SS3("".join(seq_list))

    # todo : raise 문은 예외처리로 동작가능하게끔 .. assert는 내부 정확성을 위해서?  https://google.github.io/styleguide/pyguide.html#244-decision 읽어보자.

    def vocab_check(self) -> None:
        """
        check the key error.
        """
        vocabulary = vocab.ss8_idx_vocab

        for i, current_ss8 in enumerate(self.ss8):
            if current_ss8 not in vocabulary:
                raise ValueError(
                    f'"{current_ss8}" is not existed in vocab. idx number : {i}'
                )

    # ss8
    def _ss8_idx(self, ss8: str) -> List[int]:
        """
        convert the 8-class secondary structure to one-hot vector.
        If it has unknown secondary structure, it is replaced to <unk> token.
        """
        vocabulary = vocab.ss8_idx_vocab
        unknown_token = vocabulary["X"]

        return [vocabulary.get(_ss8, unknown_token) for _ss8 in ss8]

    def _idx_ss8(self, idx: List[int]) -> List[str]:
        """
        convert the ss8 one-hot vector to 8-class secondary structure
        """
        vocabulary = vocab.idx_ss8_vocab

        return "".join([vocabulary[_idx] for _idx in idx])

    def __repr__(self):
        return self.ss8

    def __eq__(self, other):
        return self.ss8 == other
