#from collections import defaultdict

# esm -  ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
#         'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
#         'X', 'B', 'U', 'Z', 'O', '.', '-']
# 'x', 'b', 'u', 'z','O','.',-'
# todo : is gap same unk ?
_aa_vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K',
          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
          'W', 'Y', '<unk>','<pad>', '<cls>' ,'<eos>', '<mask>']
aa_idx_vocab = {token:idx for idx, token in enumerate(_aa_vocab)}

# todo : what is 'X', 'Z', 'U' symbol ?


idx_aa_vocab= {v:k for k, v in aa_idx_vocab.items()}

aa_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
           'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
           'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
           'TYR': 'Y', 'VAL': 'V', 'HSE': 'H', 'HSD': 'H', 'UNK': 'X'}
aa_dict_inv = {v:k for k,v in aa_dict.items()}


# todo : 'X', '-'
_ss3_vocab = ['H','E','C','X']
_ss8_vocab = ['H','E','-','T','S','G','B','I','X']


ss3_idx_vocab = {token:idx for idx, token in enumerate(_ss3_vocab)}
idx_ss3_vocab = {v:k for k, v in ss3_idx_vocab.items()}

ss8_idx_vocab = {token:idx for idx, token in enumerate(_ss8_vocab)}
idx_ss8_vocab = {v:k for k, v in ss8_idx_vocab.items()}

ss8_ss3_vocab = {'H':'H', 'G':'H',
                 'E':'E', 'B':'E',
                 '-':'C', 'T':'C', 'S':'C', 'I':'C',
                 'X':'X'
                 }

