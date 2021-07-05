import os
import tensorflow as tf
from pdp.file_io import load_fasta

class PretrainDataTFWriter:
    def __init__(self, path):
        self.path = path
        glob_pattern = os.path.join(path, '*.fasta')
        self.file_list = glob(glob_pattern)

    def to_tfrecord(self, tfrecord_file) -> None:
        """
        export to tfreocrd file.
        """
        # todo : parallel ?
        tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file)
        for file in tqdm.tqdm(self.file_list):
            my_fasta = load_fasta(file)
            tfrecord_writer.write(my_fasta.serialize())
        tfrecord_writer.close()