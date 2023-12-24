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
  const double3* getForces() const { return m_applied_forces; }
protected:
  int m_device_id;
  int64_t m_step;
  int64_t m_num_atoms;
  double3 *m_d_positions;
  double3 *m_applied_forces;
};

class PytorchForces: public CUDAGlobalForces {
public:
  PytorchForces(const std::string& nn_model_filename);
  void openOutputFile(const std::string& output_filename);
  void calculate() override;
  bool applyForce(double* f, size_t force_size) override;
  void requestAtoms(int numAtoms) override;
private:
  torch::Tensor m_tensor_in;
  torch::jit::script::Module m_module;
  std::unique_ptr<std::ostream> m_output_stream;
  torch::TensorOptions m_option_in;
  torch::TensorOptions m_option_force;
  size_t m_output_size;
};

#endif //EXAMPLE_APP_CUDA_TORCH_H
