import pdb
import torch

cuda = torch.device('cuda')
def my_func():
    pdb.set_trace()
    y = torch.tensor([1., 2.]).cuda()

x = torch.tensor([1., 2.], device=cuda)
my_func()
