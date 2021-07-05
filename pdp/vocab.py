#from collections import defaultdict

# todo : is gap is same unk ?
_vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K',
          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
          'W', 'Y', '-', '<unk>','<pad>', '<cls>' ,'<eos>', '<mask>']

aa2idx_vocab = {token:idx for idx, token in enumerate(_vocab)}
idx2aa_vocab= {v:k for k,v in aa2idx_vocab.items()}

aa_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
           'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
           'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V', 'HSE': 'H', 'HSD': 'H', 'UNK': 'X'}
aa_dict_inv = {v:k for k,v in aa_dict.items()}
