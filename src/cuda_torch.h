#ifndef EXAMPLE_APP_CUDA_TORCH_H
#define EXAMPLE_APP_CUDA_TORCH_H

#include "cuda_runtime.h"
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <memory>

class CUDAGlobalForces {
public:
  CUDAGlobalForces(int cuda_device_id = 0);
  virtual ~CUDAGlobalForces();
  virtual void requestAtoms(int numAtoms);
  bool updatePositions(const std::vector<double3>& pos);
  virtual bool applyForce(double* f, size_t force_size) = 0;
  virtual void calculate() = 0;
  virtual const double3* getForces() const = 0;
protected:
  int m_device_id;
  int64_t m_step;
  int64_t m_num_atoms;
  double3 *m_d_positions;
};

class PytorchForces: public CUDAGlobalForces {
public:
  PytorchForces(const std::string& nn_model_filename);
  void openOutputFile(const std::string& output_filename);
  void calculate() override;
  bool applyForce(double* f, size_t force_size) override;
  void requestAtoms(int numAtoms) override;
  const double3* getForces() const override { return reinterpret_cast<const double3*>(m_force_out.const_data_ptr()); }
private:
  torch::Tensor m_tensor_in;
  torch::jit::script::Module m_module;
  std::unique_ptr<std::ostream> m_output_stream;
  torch::TensorOptions m_option;
  size_t m_output_size;
  torch::Tensor m_force_out;
};

#endif //EXAMPLE_APP_CUDA_TORCH_H
