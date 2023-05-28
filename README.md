
## GPU tensorstore
GPU tensorstore is built based on the Apache Arrow project, which contains a set of technologies that enable big data systems to process and move data fast. [Plasma Object Store](https://github.com/apache/arrow/tree/master/cpp/src/plasma) is a subproject which is a shared-memory blob store.

The project is built in the form of pytorch extension, the implementatin detail could be found in `arrow-tensorstore/python/pyarrow/torch-extension/`

### Module Setup
```
cd arrow-tensorstore/python/pyarrow/torch-extension/tensorstore
sudo apt-get install python3-setuptools
python3 setup.py install
```

The test python script could be found in `python/pyarrow/torch-extension/test`:

- `ptr2tensor_test.py`: prototype verification of zero-copy tranformation from cuda buffer pointer to torch tensor;
- `finaltest_trch_plasma.py`: the full demo of model sharing process including extracting/replacing tensor and forward inference. 

