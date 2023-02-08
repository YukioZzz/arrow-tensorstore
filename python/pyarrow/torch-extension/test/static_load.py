import torch
import copy
from torchvision import models, transforms
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.plasma as plasma
import tensorstore_helper
import time
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

def inference_once(inf_model):
    image = Image.open('cat_224x224.jpg')
    image = transform_image(image)
    tensor = torch.from_numpy(np.array(image))
    tensor = tensor.view(-1, 3, 224, 224).cuda()
    t0 = time.time()
    result = inf_model(tensor)[0]
    t1 = time.time()
    print("inner inference elapsed:", t1-t0)
    prediction = F.softmax(result, dim=0)
    topk_vals, topk_idxs = torch.topk(prediction, 3)
    print(topk_vals, topk_idxs)
    import json
    json_file = open('imagenet_class_index.json')
    json_str = json_file.read()
    labels = json.loads(json_str)      
    data = []

    #for i in range(len(topk_idxs)):
    #    r = {"label": labels[str(topk_idxs[i].item())][1],
    #         "probability": topk_vals[i].item()}
    #    data.append(r)
    #print(data)

def checkModelSize(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size()) 
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    
    print(f'Total params: {Total_params / 1e6}M')
    print(f'Trainable params: {Trainable_params/ 1e6}M')
    print(f'Non-trainable params: {NonTrainable_params/ 1e6}M')

def initFramework():
    print("initing framework", time.time())
    #_ = torch.cuda.IntTensor(1)
    mdl = models.resnet50(weights='DEFAULT').cuda()
    inference_once(mdl)
    del mdl
    torch.cuda.empty_cache()

def checkModel(model, shared=False):
    t0 = time.time()
    if shared:
        from tensorstore import storage
        model = storage.to_device(model)
    else:
        model = model.to(device='cuda')
    t1 = time.time()
    print("to cuda elapsed:", t1-t0)
    inference_once(model)
    t2 = time.time()
    print("total inference elapsed:", t2-t1)
    checkModelSize(model)

def createModel(model_name):
    print("creating model")
    from timm import create_model
    return create_model(model_name, pretrained=True)

if __name__ == "__main__":
    initFramework()
    model_list = ["resnet152", "convnext_xlarge_in22k", "vit_huge_patch14_224_in21k"]
    model = createModel(model_list[2])
    checkModel(model,shared=True)
    time.sleep(100)
