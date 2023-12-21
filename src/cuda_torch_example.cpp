#include "cuda_torch.h"
#include "helper_cuda.h"
#include <fstream>
#include <string>
#include <regex>
#include <string_view>
#include <charconv>

// Helper function to split a line of numbers into a vector
std::vector<double3> parse_line_to_position(std::string_view s) {
  const std::regex pattern("[^[:space:]]+", std::regex::extended);
  using svregex_const_iterator = std::regex_iterator<std::string_view::const_iterator>;
  svregex_const_iterator iter(s.begin(), s.end(), pattern);
  svregex_const_iterator end;
  std::vector<double3> results;
  while (iter != end) {
    double3 data;
    if (!iter->empty()) {
      const auto tmp = s.substr(iter->position(), iter->length());
      std::from_chars(tmp.data(), tmp.data() + tmp.size(), data.x);
    }
    ++iter;
    if (!iter->empty()) {
      const auto tmp = s.substr(iter->position(), iter->length());
      std::from_chars(tmp.data(), tmp.data() + tmp.size(), data.y);
    }
    ++iter;
    if (!iter->empty()) {
      const auto tmp = s.substr(iter->position(), iter->length());
      std::from_chars(tmp.data(), tmp.data() + tmp.size(), data.z);
    }
    ++iter;
    results.push_back(data);
  }
  return results;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Invalid argument.\n";
    return 1;
  }
  PytorchForces pytorch_forces(argv[1]);
  double* d_force_applied_to_cv;
  const double force_applied_to_cv = 100.0;
  checkCudaErrors(cudaMalloc(&d_force_applied_to_cv, 1 * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_force_applied_to_cv, &force_applied_to_cv, 1 * sizeof(double), cudaMemcpyHostToDevice));
  bool first_time = true;
  pytorch_forces.openOutputFile("test.out");
  std::ifstream ifs(argv[2]);
  std::string line;
  while (std::getline(ifs, line)) {
    const auto current_atom_pos = parse_line_to_position(line);
    if (first_time) {
      pytorch_forces.requestAtoms(static_cast<int>(current_atom_pos.size()));
      first_time = false;
    }
    pytorch_forces.updatePositions(current_atom_pos);
    pytorch_forces.calculate();
    pytorch_forces.applyForce(d_force_applied_to_cv, 1);
  }
  cudaFree(d_force_applied_to_cv);
  return 0;
}
