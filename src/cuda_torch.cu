#include "cuda_torch.h"
#include "helper_cuda.h"
#include <fmt/format.h>
#include <iostream>

CUDAGlobalForces::CUDAGlobalForces(int device_id):
  m_device_id(device_id), m_step(0), m_num_atoms(0), m_d_positions(nullptr) {}

CUDAGlobalForces::~CUDAGlobalForces() {
  if (m_d_positions != nullptr) {
    checkCudaErrors(cudaFree(m_d_positions));
  }
}

void CUDAGlobalForces::requestAtoms(int numAtoms) {
  m_num_atoms = numAtoms;
  if (m_d_positions != nullptr) {
    checkCudaErrors(cudaFree(m_d_positions));
  }
  checkCudaErrors(cudaMalloc(&m_d_positions, m_num_atoms * sizeof(double3)));
  checkCudaErrors(cudaMemset(m_d_positions, 0, m_num_atoms * sizeof(double3)));
}

bool CUDAGlobalForces::updatePositions(const std::vector<double3> &pos) {
  if (pos.size() != m_num_atoms) return false;
  const double3* h_pos = pos.data();
  checkCudaErrors(cudaMemcpy(m_d_positions, h_pos, m_num_atoms * sizeof(double3), cudaMemcpyHostToDevice));
  m_step++;
  return true;
}

PytorchForces::PytorchForces(const std::string& nn_model_filename):
  CUDAGlobalForces(), m_output_stream(nullptr), m_output_size(0) {
  try {
    m_module = torch::jit::load(nn_model_filename);
    m_module.to(torch::kFloat64);
    m_module.to(at::Device("cuda"));
  } catch (const c10::Error& e) {
    throw;
  }
}

void PytorchForces::requestAtoms(int numAtoms) {
  CUDAGlobalForces::requestAtoms(numAtoms);
  m_option = torch::TensorOptions()
    .dtype(torch::kFloat64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, m_device_id)
    .requires_grad(true);
  m_tensor_in = torch::from_blob(m_d_positions, {m_num_atoms, 3}, m_option);
}

void PytorchForces::calculate() {
  std::vector<torch::jit::IValue> input_args{m_tensor_in};
  const auto value_out = m_module.get_method("calc_value")(input_args);
  auto output_dim = value_out.toTensor().sizes();
  if (output_dim.empty()) m_output_size = 1;
  else m_output_size = value_out.toTensor().sizes()[0];
  if (m_output_stream) {
    (*m_output_stream) << fmt::format(" {:12d} {:15.7e}\n", m_step, value_out.toTensor().item<double>());
  }
}

void PytorchForces::openOutputFile(const std::string &output_filename) {
  if (m_output_stream) {
    return;
  } else {
    m_output_stream = std::make_unique<std::ofstream>(output_filename);
    (*m_output_stream) << fmt::format("#{:>12s} {:>15s}\n", "step", "value");
  }
}

bool PytorchForces::applyForce(double* f, size_t force_size) {
  if (force_size != m_output_size) {
    std::cerr << fmt::format("force_size = {}, cv_size = {}\n", force_size, m_output_size);
    return false;
  }
  if (auto apply_force = m_module.find_method("apply_force")) {
    auto in_force_tensor = torch::from_blob(f, {static_cast<long>(force_size)}, m_option);
    std::vector<torch::jit::IValue> input_args{m_tensor_in, in_force_tensor};
    m_force_out = (apply_force.value())(input_args).toTensor();
  } else {
    std::cerr << "Cannot find function \"apply_force\"\n";
    return false;
  }
  return true;
}
