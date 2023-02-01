#include <torch/extension.h>
//#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
using namespace at;
using namespace std;

void checkCudaError(string func, cudaError_t err){
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed at %s with err %s\n", func, cudaGetErrorString(err));
      exit(1);
  }
}
void ptrtest(uint64_t address){
  /* for test */
  //CUdeviceptr test_addr;
  //cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&test_addr), 1000*sizeof(uint32_t));
  uint32_t buff[1000];
  cudaPointerAttributes attributes;
  checkCudaError("getAttributes", cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(address)));
  std::cout<<"ptr device:"<<attributes.type<<std::endl;

  checkCudaError("cudaMemcpy", cudaMemcpy(reinterpret_cast<void*>(buff), reinterpret_cast<void*>(address), 1000 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::cout<<"check first 1000 value:"<<std::endl;
  for(int i=0;i<1000;i++){
    std::cout<<buff[i]<<" ";
  }
}

Tensor ptr2CudaTensor(uint64_t address, pybind11::array_t<long int> sizes, pybind11::array_t<long int> strides, py::object dtype){
  // for test
  //ptrtest(address);
  // for work
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
  auto options = torch::TensorOptions().dtype(type).device(torch::kCUDA);

  auto sizesRef = torch::ArrayRef<long int>((long int*)sizes.data(), strides.size());
  auto stridesRef = torch::ArrayRef<long int>((long int*)strides.data(), strides.size());
  auto f = torch::from_blob(reinterpret_cast<void*>(address), sizesRef, stridesRef, options); 
  return f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ptr2CudaTensor", &ptr2CudaTensor, "load data from existing place without copy");
}
