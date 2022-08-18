import logging
import os

import tensorflow as tf

from pdp.data.pdb70_processor import (
    parsing_pdb70,
    PDB70_PATH,
    PDB_PATH,
    split_pdb_name,
    PDB70_Processor,
    get_exist_pdb,
)
from pdp.data.paths import get_expand_path

# todo : 1. when perform the transforming the data.
# option 1. Before transforming write training example ->  we have to transforming after parsing.
# option 2. After transforming write training example -> we need only parsing.


def write_instance_to_example_files(
    processor: PDB70_Processor, output_files: list, gzip_compress=True
):
    """Creates TF example files from processor"""

    # todo : automatic set output files -> It is comfortable that we set automatically number of output from considering the number of data, size etc.
    output_folder = os.path.dirname(output_files[0])
    os.makedirs(output_folder, exist_ok=True)

    writers = []
    for output_file in output_files:
        writers.append(
            tf.io.TFRecordWriter(output_file, options="GZIP" if gzip_compress else "")
        )

    writer_index = 0
    total_written = 0

    data_iterator = iter(processor)

    for (inst_index, afdata) in enumerate(data_iterator):
        try:
            example = afdata.get_example()
            print(inst_index, end="\r")
        except BaseException as e:
            print(e)
            print(afdata.domain_name, inst_index)

        writers[writer_index].write(example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

        if inst_index < 20:
            logging.info("*** Example ***")
            logging.info(f"name : {afdata.domain_name}, Seq : {afdata.sequence[:10]}")

    for writer in writers:
        writer.close()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    pdb_list = parsing_pdb70(PDB70_PATH)  # todo : flags?
    exist_pdb_list, absent_pdb_list = get_exist_pdb(pdb_list)

    pdb70_processor = PDB70_Processor(pdb70_list=exist_pdb_list[:50])
    num_of_output = 1  # todo : flags?
    output_files = [
        get_expand_path(f"~/.cache/tfdata/pdb/structure_data_{i}.tfrecord")
        for i in range(num_of_output)
    ]

    write_instance_to_example_files(pdb70_processor, output_files)
#
#
# data_iterator = iter(pdb70_processor)
# data_iterator._error_list
#
# for inst_index, afdata in enumerate(data_iterator):
#     print(inst_index)
#     example = afdata.get_example()
