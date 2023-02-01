// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <assert.h>
#include <signal.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <memory>
#include <thread>

#include "arrow/util/io_util.h"

#include "plasma/test_util.h"
#include "plasma/client.h"
#include "plasma/common.h"
#include "plasma/plasma.h"
#include "plasma/protocol.h"

using arrow::cuda::CudaBuffer;
using arrow::cuda::CudaBufferReader;
using arrow::cuda::CudaBufferWriter;

constexpr int kGpuDeviceNumber = 1;

namespace plasma {

namespace {

void AssertCudaRead(const std::shared_ptr<Buffer>& buffer,
                    const std::vector<uint8_t>& expected_data) {
  std::shared_ptr<CudaBuffer> gpu_buffer;
  const size_t data_size = expected_data.size();

  gpu_buffer =  CudaBuffer::FromBuffer(buffer).ValueOrDie();

  CudaBufferReader reader(gpu_buffer);
  std::vector<uint8_t> read_data(data_size);
  auto result = reader.Read(data_size, read_data.data());

  for (size_t i = 0; i < data_size; i++) {
    std::cout<< "Get: "<<read_data[i] << " expected: " <<expected_data[i] <<std::endl;
  }
}

}  // namespace

int main(){
  PlasmaClient client_;
  ARROW_CHECK_OK(client_.Connect("/tmp/plasma", ""));
  ObjectID object_id = random_object_id();
  std::vector<ObjectBuffer> object_buffers;

  uint8_t data[] = {4, 5, 3, 1};
  int64_t data_size = sizeof(data);
  uint8_t metadata[] = {42};
  int64_t metadata_size = sizeof(metadata);
  std::shared_ptr<Buffer> data_buffer;
  std::shared_ptr<CudaBuffer> gpu_buffer;
  ARROW_CHECK_OK(client_.Create(object_id, data_size, metadata, metadata_size,
                                &data_buffer, kGpuDeviceNumber));
  gpu_buffer =  CudaBuffer::FromBuffer(data_buffer).ValueOrDie();
  CudaBufferWriter writer(gpu_buffer);
  ARROW_CHECK_OK(writer.Write(data, data_size));
  ARROW_CHECK_OK(client_.Seal(object_id));

  object_buffers.clear();
  ARROW_CHECK_OK(client_.Get({object_id}, -1, &object_buffers));
  std::cout<<object_buffers.size()<<std::endl;
  std::cout<<object_buffers[0].device_num <<std::endl;
  // Check data
  AssertCudaRead(object_buffers[0].data, {4, 5, 3, 1});
  // Check metadata
  AssertCudaRead(object_buffers[0].metadata, {42});
  return 0;
}
}
