#include "cuda_torch.h"
#include "helper_cuda.h"
#include <fmt/format.h>
#include <iostream>

CUDAGlobalForces::CUDAGlobalForces(int device_id):
  m_device_id(device_id), m_step(0), m_num_atoms(0), m_d_positions(nullptr), m_applied_forces(nullptr) {}

CUDAGlobalForces::~CUDAGlobalForces() {
  if (m_d_positions != nullptr) {
    checkCudaErrors(cudaFree(m_d_positions));
  }
  if (m_applied_forces != nullptr) {
    checkCudaErrors(cudaFree(m_applied_forces));
  }
}

void CUDAGlobalForces::requestAtoms(int numAtoms) {
  m_num_atoms = numAtoms;
  if (m_d_positions != nullptr) {
    checkCudaErrors(cudaFree(m_d_positions));
  }
  checkCudaErrors(cudaMalloc(&m_d_positions, 3 * m_num_atoms * sizeof(double)));
  checkCudaErrors(cudaMemset(m_d_positions, 0, 3 * m_num_atoms * sizeof(double)));
  if (m_applied_forces != nullptr) {
    checkCudaErrors(cudaFree(m_applied_forces));
  }
  checkCudaErrors(cudaMalloc(&m_applied_forces, 3 * m_num_atoms * sizeof(double)));
  checkCudaErrors(cudaMemset(m_applied_forces, 0, 3 * m_num_atoms * sizeof(double)));
}

bool CUDAGlobalForces::updatePositions(const std::vector<double3> &pos) {
  if (pos.size() != m_num_atoms) return false;
  std::vector<double> pos_t(m_num_atoms * 3);
  for (size_t i = 0; i < m_num_atoms; ++i) {
    pos_t[i] = pos[i].x;
    pos_t[i + m_num_atoms] = pos[i].y;
    pos_t[i + m_num_atoms * 2] = pos[i].z;
  }
  checkCudaErrors(cudaMemcpy(m_d_positions, pos_t.data(), 3 * m_num_atoms * sizeof(double), cudaMemcpyHostToDevice));
  m_step++;
  return true;
}

PytorchForces::PytorchForces(const std::string& nn_model_filename):
  CUDAGlobalForces(), m_output_stream(nullptr), m_output_size(0) {
  try {
    m_module = torch::jit::load(nn_model_filename);
    m_module.to(torch::kFloat64);
    m_module.to(at::Device("cuda"));
    std::vector<torch::jit::IValue> input_args{};
    const auto value_out = m_module.get_method("output_dim")(input_args);
    m_output_size = value_out.toInt();
  } catch (const c10::Error& e) {
    throw;
  }
}

void PytorchForces::requestAtoms(int numAtoms) {
  CUDAGlobalForces::requestAtoms(numAtoms);
  m_option_in = torch::TensorOptions()
    .dtype(torch::kFloat64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, m_device_id)
    .requires_grad(true);
  m_option_force = torch::TensorOptions()
    .dtype(torch::kFloat64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, m_device_id)
    .requires_grad(false);
  m_tensor_in = torch::from_blob(m_d_positions, {3, numAtoms}, m_option_in).transpose(0, 1);
}

void PytorchForces::calculate() {
  std::vector<torch::jit::IValue> input_args{m_tensor_in};
  const auto value_out = m_module.get_method("calc_value")(input_args).toTensor();
  std::vector<double> host_out(value_out.numel());
  value_out.contiguous();
  cudaMemcpy(host_out.data(), value_out.const_data_ptr(), m_output_size * sizeof(double), cudaMemcpyDeviceToHost);
  (*m_output_stream) << fmt::format(" {:12d} {:15.7e}\n", m_step, fmt::join(host_out, ""));
}

void PytorchForces::openOutputFile(const std::string &output_filename) {
  if (m_output_stream) {
    return;
  } else {
    m_output_stream = std::make_unique<std::ofstream>(output_filename);
    std::vector<torch::jit::IValue> input_args{};
    const auto cv_names = m_module.get_method("cv_names")(input_args).toList().vec();
    (*m_output_stream) << fmt::format("#{:>12s}", "step");
    for (size_t i = 0; i < cv_names.size(); ++i){
      (*m_output_stream) << fmt::format(" {:>15s}", cv_names[i].toStringRef());
    }
    (*m_output_stream) << std::endl;
  }
}

bool PytorchForces::applyForce(double* f, size_t force_size) {
  if (force_size != m_output_size) {
    std::cerr << fmt::format("force_size = {}, cv_size = {}\n", force_size, m_output_size);
    return false;
  }
  if (auto apply_force = m_module.find_method("apply_force")) {
    auto in_force_tensor = torch::from_blob(f, {static_cast<long>(force_size)}, m_option_in);
    std::vector<torch::jit::IValue> input_args{m_tensor_in, in_force_tensor};
    // https://stackoverflow.com/questions/71790378/assign-memory-blob-to-py-torch-output-tensor-c-api
    torch::from_blob(m_applied_forces, {3, m_num_atoms}, m_option_force).transpose(0, 1) = (apply_force.value())(input_args).toTensor();
  } else {
    std::cerr << "Cannot find function \"apply_force\"\n";
    return false;
  }
  return true;
}
