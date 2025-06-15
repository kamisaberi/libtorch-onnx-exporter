#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include "onnx.proto3.pb.h"

using json = nlohmann::json;

struct ManualTensor {
    std::vector<int64_t> dims;
    std::vector<float> data;
};

// --- NEW HELPER: Transpose a 2D matrix ---
ManualTensor transpose(const ManualTensor& tensor) {
    if (tensor.dims.size() != 2) {
        throw std::runtime_error("Transpose only supports 2D tensors.");
    }
    int64_t rows = tensor.dims[0];
    int64_t cols = tensor.dims[1];

    ManualTensor transposed_tensor;
    transposed_tensor.dims = {cols, rows};
    transposed_tensor.data.resize(rows * cols);

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            transposed_tensor.data[j * rows + i] = tensor.data[i * cols + j];
        }
    }
    return transposed_tensor;
}

// --- NEW REAL WEIGHTS READER ---
std::map<std::string, ManualTensor> read_weights_file(const std::string& filepath, const std::vector<std::string>& param_names) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open weights file: " + filepath);
    }

    std::map<std::string, ManualTensor> weights;

    for (const auto& name : param_names) {
        ManualTensor tensor;

        // 1. Read number of dimensions
        int64_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
        if (file.gcount() != sizeof(num_dims)) throw std::runtime_error("Failed to read num_dims for " + name);

        // 2. Read the dimensions
        tensor.dims.resize(num_dims);
        file.read(reinterpret_cast<char*>(tensor.dims.data()), num_dims * sizeof(int64_t));
        if (file.gcount() != num_dims * sizeof(int64_t)) throw std::runtime_error("Failed to read dims for " + name);

        // 3. Read the raw data
        int64_t num_elements = 1;
        for (int64_t dim : tensor.dims) {
            num_elements *= dim;
        }
        tensor.data.resize(num_elements);
        file.read(reinterpret_cast<char*>(tensor.data.data()), num_elements * sizeof(float));
        if (file.gcount() != num_elements * sizeof(float)) throw std::runtime_error("Failed to read data for " + name);

        weights[name] = tensor;
    }

    return weights;
}

// --- (add_initializer function is the same as before) ---
void add_initializer(onnx::GraphProto* graph, const std::string& name, const ManualTensor& tensor) {
    auto* initializer = graph->add_initializer();
    initializer->set_name(name);
    initializer->set_data_type(onnx::TensorProto_DataType_FLOAT);
    for (int64_t dim : tensor.dims) {
        initializer->add_dims(dim);
    }
    initializer->set_raw_data(tensor.data.data(), tensor.data.size() * sizeof(float));
}

int main() {
    const std::string arch_path = "model_arch.json";
    const std::string weights_path = "model_weights.bin";
    const std::string onnx_output_path = "model_manual_export.onnx";

    std::cout << "--- Custom C++ ONNX Exporter (Reading .bin) ---" << std::endl;

    std::ifstream arch_file(arch_path);
    json arch = json::parse(arch_file);
    std::cout << "Loaded architecture from " << arch_path << std::endl;

    // Use the parameter order from the JSON to read the binary file correctly
    std::vector<std::string> param_names = arch["param_order"];
    auto weights = read_weights_file(weights_path, param_names);
    std::cout << "Loaded " << weights.size() << " tensors from " << weights_path << std::endl;

    // ... (The rest of the main function is identical to the previous, correct version) ...

    onnx::ModelProto model_proto;
    model_proto.set_ir_version(9);
    model_proto.set_producer_name("Corrected Manual Exporter");
    model_proto.add_opset_import()->set_version(14);

    onnx::GraphProto* graph = model_proto.mutable_graph();
    graph->set_name("main_graph");

    auto* input_info = graph->add_input();
    std::string current_tensor_name = "input";
    input_info->set_name(current_tensor_name);
    auto* input_type = input_info->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    input_type->mutable_shape()->add_dim()->set_dim_param("batch_size");
    input_type->mutable_shape()->add_dim()->set_dim_value(10);

    for (const auto& layer : arch["layers"]) {
        std::string layer_name = layer["name"];
        std::string layer_type = layer["type"];

        if (layer_type == "Linear") {
            std::string weight_name = layer["params"][0];
            std::string bias_name = layer["params"][1];
            std::string matmul_out_name = layer_name + "_matmul_out";
            std::string add_out_name = layer_name + "_add_out";

            ManualTensor transposed_weight = transpose(weights.at(weight_name));

            add_initializer(graph, weight_name, transposed_weight);
            add_initializer(graph, bias_name, weights.at(bias_name));

            auto* matmul_node = graph->add_node();
            matmul_node->set_op_type("MatMul");
            matmul_node->add_input(current_tensor_name);
            matmul_node->add_input(weight_name);
            matmul_node->add_output(matmul_out_name);

            auto* add_node = graph->add_node();
            add_node->set_op_type("Add");
            add_node->add_input(matmul_out_name);
            add_node->add_input(bias_name);
            add_node->add_output(add_out_name);

            current_tensor_name = add_out_name;
        } else if (layer_type == "ReLU") {
            std::string relu_out_name = layer_name + "_out";
            auto* relu_node = graph->add_node();
            relu_node->set_op_type("Relu");
            relu_node->add_input(current_tensor_name);
            relu_node->add_output(relu_out_name);
            current_tensor_name = relu_out_name;
        }
    }

    auto* output_info = graph->add_output();
    output_info->set_name(current_tensor_name);
    auto* output_type = output_info->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    output_type->mutable_shape()->add_dim()->set_dim_param("batch_size");
    output_type->mutable_shape()->add_dim()->set_dim_value(5);

    std::ofstream out_stream(onnx_output_path, std::ios::out | std::ios::binary);
    model_proto.SerializeToOstream(&out_stream);

    std::cout << "Successfully created ONNX model at: " << onnx_output_path << std::endl;

    return 0;
}