# nohup plasma_store -m 1000000000 -s /tmp/plasma &
# python -i /arrow/torch/ptr2tensor_test.py
import pyarrow as pa
import numpy as np
import pyarrow.plasma as plasma
import torch
torch.cuda.ByteTensor(1)#avoid lazy init

tensor = torch.IntTensor(np.ones(1000, dtype=np.int32))
buf = pa.py_buffer(tensor.numpy())#serialize and revmove the metadata but keep dtype

client = plasma.connect("/tmp/plasma")
obj_id = plasma.ObjectID(20*b'k')
def getObj(obj_id):
    try:
        obj_id = client.put(buf, object_id=obj_id, device_num=1)
    except plasma.PlasmaObjectExists:
        print("Tensor already exists")
        pass
    tensorref = client.get(obj_id, device_num=1)
    print("tnsr got from store addr:",tensorref.address)
    #print(tensorref.to_pybytes())
    print(tensorref.address)
    import tensorstore_helper
    tnr = tensorstore_helper.ptr2CudaTensor(tensorref.address, tensor.size(), tensor.stride(), tensor.dtype)
    print("TorchTensor addr:", tnr.data_ptr())
    print(tnr)
getObj(obj_id)
getObj(obj_id)
getObj(obj_id)
import time
time.sleep(100000)
