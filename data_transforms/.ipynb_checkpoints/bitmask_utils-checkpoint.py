import numpy as np
import random
import torch

if torch.cuda.is_available():
  pt_device = "cuda"
else:
  pt_device = "cpu"

#assumes operation on flattened matrix

# bitmask_utils.py

def to_bitmask(M):
    """
    Returns the bitmask sparse format of matrix 'M'
    :param M: original weight matrix M
    """
    wmb = (M != 0).float()
    data = M[torch.nonzero(M, as_tuple=True)]

    print(f'wmb shape: {wmb.shape}')
    print(f'data shape: {data.shape}')

    return wmb, data

def from_bitmask(wmb, data):
    """
    Translate bitmask encoded data back to original matrix M
    :param wmb: weight mask bits -- bit vectors to indicate non-zero values in original weight matrix M
    :param data: Data corresponding to the non-zero elements of matrix M
    """
    M = torch.zeros(wmb.size()[0], device=pt_device, dtype=torch.float32)

    nonzero_indices = (wmb == 1).nonzero(as_tuple=True)[0]
    M[nonzero_indices] = data[nonzero_indices]

    return M


def encoded_capacity(wmb, data, num_bits):
  """
  Returns the total capacity in bits of the bitmask + data

  :param wmb: weight mask bits -- bit vectors to indicate non-zero values in original weight matrix M

  :param data: Data corresponding to the non-zero elements of matrix M

  :param num_bits: number of bits per data value to use

  """
  return (data.size()[0]*num_bits + wmb.size()[0])