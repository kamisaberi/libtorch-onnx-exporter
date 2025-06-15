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

// Function to write a single tensor to our custom binary stream
void write_tensor(std::ofstream& stream, const torch::Tensor& tensor) {
    torch::Tensor t = tensor.contiguous();

    // 1. Write number of dimensions
    int64_t num_dims = t.dim();
    stream.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));

    // 2. Write the dimensions
    stream.write(reinterpret_cast<const char*>(t.sizes().data()), num_dims * sizeof(int64_t));

    // 3. Write the raw data
    stream.write(reinterpret_cast<const char*>(t.data_ptr<float>()), t.numel() * sizeof(float));
}

int main() {
    const std::string arch_path = "model_arch.json";
    const std::string weights_path = "model_weights.bin"; // Using a new extension

    auto model = std::make_shared<SimpleNet>();
    std::cout << "--- C++ Model Serializer (Custom Binary Format) ---" << std::endl;

    // --- Save architecture to JSON ---
    json model_arch;
    model_arch["input_shape"] = {1, 10};
    model_arch["output_shape"] = {1, 5};

    std::vector<std::string> param_names;
    for (const auto& pair : model->named_parameters()) {
        param_names.push_back(pair.key());
    }
    model_arch["param_order"] = param_names; // Store the exact order of weights

    model_arch["layers"] = {
        {{"name", "fc1"}, {"type", "Linear"}, {"params", {"fc1.weight", "fc1.bias"}}},
        {{"name", "relu1"}, {"type", "ReLU"}, {"params", json::array()}},
        {{"name", "fc2"}, {"type", "Linear"}, {"params", {"fc2.weight", "fc2.bias"}}}
    };
    std::ofstream arch_file(arch_path);
    arch_file << model_arch.dump(4);
    arch_file.close();
    std::cout << "Saved architecture to " << arch_path << std::endl;

    // --- Save parameters to our custom binary file ---
    std::ofstream weights_file(weights_path, std::ios::binary);
    if (!weights_file) {
        std::cerr << "Error opening weights file for writing: " << weights_path << std::endl;
        return 1;
    }

    auto named_params = model->named_parameters();
    for (const auto& pair : named_params) {
        write_tensor(weights_file, pair.value());
    }
    weights_file.close();

    std::cout << "Saved weights to custom binary file: " << weights_path << std::endl;

    return 0;
}