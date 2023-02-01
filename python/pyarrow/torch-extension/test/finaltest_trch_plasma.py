import torch
import copy
from torchvision import models, transforms
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.plasma as plasma
import tensorstore_helper
client = plasma.connect("/tmp/plasma")

def extract_tensors(m: torch.nn.Module): 
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})
    
    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)   

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors

def toplasma(name, tensor):
    # tensor must be of type torch.xxxTensor
    print("input tensor shape:", tensor.size())
    buf = pa.py_buffer(tensor.numpy())
    import hashlib;
    hashedname = hashlib.sha1(name.encode()).digest();
    obj_id = plasma.ObjectID(hashedname)
    try:
        obj_id = client.put(buf, object_id=obj_id, device_num=1)
        #obj_id = client.put(buf, device_num=1)
    except plasma.PlasmaObjectExists:
        print("Tensor already exists")
        pass# continue, use obj_id to get object directly
    tensorref = client.get(obj_id, device_num=1)
    tnr = tensorstore_helper.ptr2CudaTensor(tensorref.address, tensor.size(), tensor.stride(), tensor.dtype)
    print("tensor loaded from tensorstore with size:", tnr.size())
    print(tnr)
    return tnr


def replace_tensors(m: torch.nn.Module, tensors):
    """
    Restore the tensors that extract_tensors() stripped out of a
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    for named_modules, tensor_dict in zip(m.named_modules(), tensors):
        modname, module = named_modules
        # There are separate APIs to set parameters and buffers.
        for tnsrname, array in tensor_dict["params"].items():
            #param = torch.as_tensor(array).cuda()
            param = toplasma(m.__class__.__name__+'.'+modname+'.'+tnsrname, torch.as_tensor(array))
            module.register_parameter(tnsrname, torch.nn.Parameter(param))
        for tnsrname, array in tensor_dict["buffers"].items():
            #param = torch.as_tensor(array).cuda()
            param = toplasma(m.__class__.__name__+'.'+modname+'.'+tnsrname, torch.as_tensor(array))
            module.register_buffer(tnsrname, param)
resnet = models.resnet152(pretrained=True)
cpy, tnsr = extract_tensors(resnet)
obj_id = client.put(tnsr)
weights = client.get(obj_id)
replace_tensors(cpy, weights)

from PIL import Image
import numpy as np
def transform_image(image):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    resnet_transform = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])
    if image.mode != "RGB":
        image = image.convert("RGB")

    return resnet_transform(image)
image = Image.open('cat_224x224.jpg')
image = transform_image(image)
tensor = torch.from_numpy(np.array(image))
tensor = tensor.view(-1, 3, 224, 224).cuda()

def infer(inf_model):
    result = inf_model(tensor)[0] 
    prediction = F.softmax(result, dim=0)
    topk_vals, topk_idxs = torch.topk(prediction, 3)
    print(topk_vals, topk_idxs)
    import json
    json_file = open('imagenet_class_index.json')
    json_str = json_file.read()
    labels = json.loads(json_str)      
    data = []

    for i in range(len(topk_idxs)):
        r = {"label": labels[str(topk_idxs[i].item())][1],
             "probability": topk_vals[i].item()}
        data.append(r)
    print(data)

#check
#for name, param in cpy.named_parameters():
#    print(name, param.shape, type(param), param.is_cuda)

infer(cpy)
