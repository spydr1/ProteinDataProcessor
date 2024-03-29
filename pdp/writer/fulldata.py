import logging
import os
import tensorflow as tf
import tqdm

from esm.data import ESMStructuralSplitDataset
from pdp.data.feature import SS8
from pdp.data.fulldata import Fulldata


# Fulldata를 받게끔 ..?
# todo : 이것을 fulldata라고 추상적으로 생각할까 esmdata라고 생각할까? 추상적이지 않다면 esm만을 읽어오기 위해 설계된 것은 이상한것 같아.
class FulldataWriter:
    def __init__(
        self,
        split_level="superfamily",
        cv_partition="4",
        split="train",
    ):

        esm_structural_train = ESMStructuralSplitDataset(
            split_level=split_level,
            cv_partition=cv_partition,
            split=split,
            root_path=os.path.expanduser("~/.cache/torch/data/esm"),
            download=True,
        )
        self.split_level = split_level
        self.cv_partition = cv_partition
        self.split = split
        # if shuffle:
        #     random.seed(seed)
        #     random.shuffle(esm_structural_train.names)

        self.data = esm_structural_train

    # todo : 이미 전처리 되어있어서 shuffle 필요 없을 것 같다. -> 확인 해보자 이미 섞인 걸까 ?
    # data -> list
    def to_tfrecord(
        self,
        tfrecord_dir="~/.cache/tfdata/fulldata/",
        sequence_length=1024,
    ) -> None:

        """
        export to tfreocrd file.
        """
        tfrecord_path = os.path.join(
            tfrecord_dir, f"{self.split_level}/{self.cv_partition}.tfrecord"
        )
        tfrecord_path = os.path.expanduser(tfrecord_path)
        dirname = os.path.dirname(tfrecord_path)

        if os.path.exists(tfrecord_path):
            raise OSError(f"{tfrecord_path} is already exists")
        os.makedirs(dirname, exist_ok=True)
        record_option = tf.io.TFRecordOptions(compression_type="GZIP")
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_path, record_option)

        # todo : 이것도 병렬처리를 해놓는게 좋으려나?
        # todo : dihedral angle 추가할지 말지 (다음 버전에 추가해도 될것 같기도? - 일단은 필요없음)
        # 이 부분이 사라지고 fulldata로 제공되어야한다.
        for i, data in enumerate(tqdm.tqdm(self.data)):
            my_fulldata = Fulldata(
                name=self.data.names[i],
                aa=data["seq"],
                coords=data["coords"],
                ss8=SS8(data["ssp"]),
                dist=data["dist"],
            )
            # seq has two additional token, '<cls>' ,'<eos>'
            if (
                len(my_fulldata.aa) + 2 < sequence_length
            ):  # todo : sequence_length, 기본값 설정 1024로 ?
                tfrecord_writer.write(my_fulldata.serialize())

        else:
            tfrecord_writer.close()


def esm2tfdata(tfrecord_dir="~/.cache/tfdata/fulldata/", rewrite=False) -> None:

    tfrecord_path = os.path.expanduser(tfrecord_dir)

    if rewrite:
        import shutil

        shutil.rmtree(tfrecord_path, ignore_errors=True)

    for split_level in ["family", "superfamily", "fold"]:
        for cv_partition in ["0", "1", "2", "3", "4"]:
            logging.info(f"{split_level} {cv_partition}")
            print(f"{split_level} {cv_partition}")
            writer = FulldataWriter(
                split_level=split_level, cv_partition=cv_partition, split="valid"
            )
            writer.to_tfrecord(tfrecord_dir=tfrecord_dir)


if __name__ == "__main__":
    esm2tfdata(rewrite=True)
