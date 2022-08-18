import os


def get_expand_path(path):
    """Convert  "~/path" to "/user/path" """
    return os.path.expanduser(path)


PDB_PATH = get_expand_path("~/.cache/pdb")
PDB70_PATH = get_expand_path("~/.cache/pdb70/pdb70_clu.tsv")
