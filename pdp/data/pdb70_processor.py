import os
import logging
import gzip
from typing import List
from enum import Enum

import tensorflow as tf

from pdp.data import afdata
from pdp.data.paths import get_expand_path, PDB_PATH, PDB70_PATH
from pdp.data.helper import AbstractProcessor


class ParserType(Enum):
    PDB_FILES = 1
    PDB70 = 2


def parsing_pdb70(pdb70_clu_path=PDB70_PATH):
    """
    Get the representative pdb with chain from PDB70 clustering.
    """
    representative = []
    with open(pdb70_clu_path) as file_obj:
        for line in file_obj.readlines():
            rep, pdb = line.split("\t")
            representative.append(rep)

    return list(set(representative))


# multiple files -> read by iterator.
# Ref DataProcessor


class PDB70_Processor(AbstractProcessor):
    def __init__(
        self,
        pdb70_list=parsing_pdb70(PDB70_PATH),
    ):
        exist_pdb_list, absent_pdb_list = get_exist_pdb(pdb70_list)

        logging.info(
            f"number of exist pdb : {len(exist_pdb_list)} number of absent pdb {len(absent_pdb_list)}"
        )

        self._index = 0
        self._pdb_name_list = iter(exist_pdb_list)
        self._error_list = dict()

    def __iter__(self) -> afdata.AFData:
        return self

    def _file_open(self, file):
        if "gz" in file:
            file_obj = gzip.open(file, mode="rt")
        else:
            file_obj = open(file)

            # above code is equal to below code.
            # file_obj = gzip.open(file, mode="rt") if "gz" in file else open(file)

        return file_obj

    def __next__(self) -> afdata.AFData:
        pdb_name = next(self._pdb_name_list)

        # todo : skip or not ?
        only_name, chain = split_pdb_name(pdb_name)
        file_path = get_pdb_path(only_name)

        # if pdb is not exist, skip
        if os.path.exists(file_path) is False:
            msg = f"{file_path} is not exist"
            self._error_list[pdb_name] = msg
            logging.info(msg)
            return self.__next__()

        file_obj = self._file_open(file_path)
        with file_obj:
            pdb_str = file_obj.read()

            # Only support single model in "from_pdb_string" function
            try:
                protein_data = afdata.from_pdb_string(
                    pdb_str, chain
                )  # todo : set the rule for choosing chain 1. all chain 2. preset specific chain
            except BaseException as e:
                self._error_list[pdb_name] = e
                return self.__next__()

        return protein_data


def split_pdb_name(pdb_name: str):
    """Get only pdb name without chain"""
    name, chain = pdb_name.split("_")
    return name.lower(), chain


def download_pdb():
    # todo : it is running background, so it need progressbar.
    # rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp/data/structures/divided/pdb/ ~/.cache/pdb2 > ~/.cache/pdb_log
    mirrordir = PDB_PATH
    logfile = get_expand_path("~/.cache/pdb_log")
    server = "rsync.wwpdb.org::ftp"
    port = 33444

    cmd = f"rsync -rlpt -v -z --delete --port={port} {server}/data/structures/divided/pdb/ {mirrordir}2 > {logfile}"

    os.system(cmd)


def get_pdb_path(only_name: str) -> str:
    """
    Get the pdb file path corresponding name.
    Args :
        only_name: pdb name without chain

    Return : pdb file path
    """
    if "_" in only_name:
        raise Exception(
            f"It seems that your input include chain name. "
            f"Please Check your input : {only_name}"
        )

    only_name = only_name.lower()
    folder_name = only_name[1:3]
    file_name = f"pdb{only_name}.ent.gz"

    path = f"{PDB_PATH}/{folder_name}/{file_name}"
    return path


# output_files : list, gzip_compress=True


def get_exist_pdb(pdb_list=parsing_pdb70(PDB70_PATH)):
    exist_pdb_list = []
    absent_pdb_list = []
    for pdb_name in pdb_list:
        only_name, chain = split_pdb_name(pdb_name)
        path = get_pdb_path(only_name)
        if os.path.exists(path):
            exist_pdb_list.append(pdb_name)
        else:
            absent_pdb_list.append(pdb_name)

    return exist_pdb_list, absent_pdb_list


# for pdb in parsing_pdb70():
#     if "1HDP" in pdb:
#         print(pdb)
#         break
# path = get_pdb_path("1hdp")
# file_obj = gzip.open(path, mode="rt")
# protein_data = afdata.from_pdb_string(file_obj.read(), "A")
# protein_data
# exam = protein_data.get_example()
# protein_data.resolution
