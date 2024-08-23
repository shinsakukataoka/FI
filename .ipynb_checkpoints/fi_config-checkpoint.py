import torch
import numpy as np

# define different NVM fault models
nvm_model = 'fe_m_fe_ss_hzo'

# provide paths to fault model distributions stored in nvm_data directory
# for more information, please see nvmexplorer.seas.harvard.edu
nvm_dict = {'rram_mlc'  : 'mlc_rram_args.p',
            'sample_mlc' : 'mlc_sample_args.p',
            'fe_de_fe_sl_hzo' : 'fe_de_fe_sl_hzo.p',
            'fe_de_fe_ss_hzo' : 'fe_de_fe_ss_hzo.p',
            'fe_m_fe_ss_hzo' : 'fe_m_fe_ss_hzo.p'}

# optional print statements during nvmFI execution
Debug=True
 

if torch.cuda.is_available():
  pt_device = "cuda"
  if Debug:
    print("CUDA is available")
else:
  pt_device = "cpu"
