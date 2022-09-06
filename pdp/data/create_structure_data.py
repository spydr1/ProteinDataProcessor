import logging
import os

import tensorflow as tf
import gc
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
    count = 0

    error_dict = dict()
    while True:
        try:
            afdata = next(processor)
            example = afdata.get_example()
            writers[writer_index].write(example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)
            count += 1
            print(f"number of data : {count}")
        except StopIteration:
            break
        except BaseException as e:
            name, *args = e.args
            error_dict.update({name: args})

        if count < 20:
            logging.info("*** Example ***")
            logging.info(f"name : {afdata.domain_name}, Seq : {afdata.sequence[:10]}")

    for writer in writers:
        writer.close()
    logging.info("Wrote %d total instances", count)

    with open(get_expand_path("~/.cache/tfdata/pdb_error.log"), mode="w") as fileobj:
        for k, v in error_dict.items():
            fileobj.write(f"{k}, {v} \n")

    # todo : write final data list


from time import time

if __name__ == "__main__":
    pdb_list = parsing_pdb70(PDB70_PATH)  # todo : flags?
    exist_pdb_list, absent_pdb_list = get_exist_pdb(pdb_list)

    start_t = time()
    pdb70_processor = PDB70_Processor(pdb70_list=exist_pdb_list)
    num_of_output = 50  # todo : flags?
    output_files = [
        get_expand_path(f"~/.cache/tfdata/pdb/structure_data_{i}.tfrecord")
        for i in range(num_of_output)
    ]

    write_instance_to_example_files(pdb70_processor, output_files)

    print(time() - start_t)
#
#
# data_iterator = iter(pdb70_processor)
# data_iterator._error_list
#
# for inst_index, afdata in enumerate(data_iterator):
#     print(inst_index)
#     example = afdata.get_example()
