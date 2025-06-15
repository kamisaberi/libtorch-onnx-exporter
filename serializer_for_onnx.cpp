#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using json = nlohmann::json;

// Model definition
struct SimpleNet : torch::nn::Module {
    SimpleNet() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 32));
        relu1 = register_module("relu1", torch::nn::ReLU());
        fc2 = register_module("fc2", torch::nn::Linear(32, 5));
    }
    torch::Tensor forward(torch::Tensor x) { return fc2(relu1(fc1(x))); }
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::ReLU relu1{nullptr};
};

int main() {
    const std::string arch_path = "model_arch.json";
    const std::string weights_path = "model_weights.pt";

    auto model = std::make_shared<SimpleNet>();
    std::cout << "--- C++ Model Serializer (for ONNX export) ---" << std::endl;

    // --- Save architecture to JSON ---
    json model_arch;
    model_arch["input_shape"] = {1, 10}; // Batch, Features
    model_arch["output_shape"] = {1, 5};
    // Let's store the parameter names in the JSON to help the exporter
    model_arch["layers"] = {
        {{"name", "fc1"}, {"type", "Linear"}, {"params", {"fc1.weight", "fc1.bias"}}},
        {{"name", "relu1"}, {"type", "ReLU"}, {"params", json::array()}}, // No params for ReLU
        {{"name", "fc2"}, {"type", "Linear"}, {"params", {"fc2.weight", "fc2.bias"}}}
    };
    std::ofstream arch_file(arch_path);
    arch_file << model_arch.dump(4);
    arch_file.close();
    std::cout << "Saved architecture to " << arch_path << std::endl;

    // --- FIX: Save parameters as a simple vector of tensors ---
    auto named_params = model->named_parameters();
    std::vector<torch::Tensor> params_vec;
    for (const auto& pair : named_params) {
        params_vec.push_back(pair.value());
    }

    // Save the vector of tensors. This is a supported operation.
    torch::save(params_vec, weights_path);
    std::cout << "Saved weights vector to " << weights_path << std::endl;

    return 0;
}