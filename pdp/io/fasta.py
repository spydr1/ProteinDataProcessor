import os
from pdp.data import AminoAcid, Fasta

def load_fasta(fasta_file)-> Fasta:
    basename = os.path.basename(fasta_file)
    fasta_name = os.path.splitext(basename)[0]

    with open(fasta_file) as file_obj:
        _ = file_obj.readline().strip()
        seq = file_obj.readline().strip()

    new_fasta = Fasta(fasta_name,seq)
    return new_fasta
