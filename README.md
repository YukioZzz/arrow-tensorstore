
## GPU tensorstore
GPU tensorstore is built based on the Apache Arrow project, which contains a set of technologies that enable big data systems to process and move data fast. [Plasma Object Store](https://github.com/apache/arrow/tree/master/cpp/src/plasma) is a subproject which is a shared-memory blob store.

The project is built in the form of pytorch extension, the implementatin detail could be found in `arrow-tensorstore/python/pyarrow/torch-extension/`

### Quick start with Docker
Note, you have to clone the repo independently, but not as a submodule to keep `.git` in the project, otherwise the compilation might fail.

1. Docker build: `docker build -t tensorstore:latest .` 
2. Docker run: `docker run -it --name cudashare --shm-size 1000000000 tensorstore:latest /bin/bash`; here, the arg `shm-size` is used to specify the shared memory used by the docker. As most models are large, we have to reserve enough of it
3. Open another terminal to run plasma server: 
  - `docker exec -it cudashare /bin/bash`
  - `plasma_store -m 1000000000 -s /tmp/plasma`. You can also run it in the background: `nohup plasma_store -m 1000000000 -s /tmp/plasma &`. But with running it in the foreground, we can see the detail logs of the plasma server.
4. In the original terminal, run the test with the python client: 
  - `cd /arrow/python/pyarrow/torch-extension/test`
  - `python3 ptr2tensor_test.py` or `python3 finaltest_trch_plasma.py`

The two tests are:
- `ptr2tensor_test.py`: prototype verification of zero-copy tranformation from cuda buffer pointer to torch tensor;
- `finaltest_trch_plasma.py`: the full demo of model sharing process including extracting/replacing tensor and forward inference. 

