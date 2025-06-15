#include <onnxruntime/onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric> // For std::iota

// Helper function to print a vector
template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& title) {
    std::cout << title;
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "\n--- C++ ONNX Importer and Inference (Maximum Compatibility API) ---" << std::endl;

    // Use a standard C-style string for the path.
    const char* model_path = "model_manual_export.onnx";

    // 1. Initialize ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Importer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); // Good for consistency

    // 2. Create the Inference Session
    std::cout << "Loading model from: " << model_path << std::endl;
    Ort::Session session(env, model_path, session_options);
    std::cout << "Model loaded successfully." << std::endl;

    // 3. Get Model Input and Output Details
    Ort::AllocatorWithDefaultOptions allocator;

    // --- Input Details ---
    size_t num_input_nodes = session.GetInputCount();
    if (num_input_nodes == 0) {
        throw std::runtime_error("Model has no inputs.");
    }

    // Get input name using the allocated string pointer method
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();
    std::cout << "Input Name: " << input_name << std::endl;

    // Get input shape
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    input_dims[0] = 1; // Set the dynamic batch size to 1 for this run
    print_vector(input_dims, "Input Shape: ");

    // --- Output Details ---
    size_t num_output_nodes = session.GetOutputCount();
    if (num_output_nodes == 0) {
        throw std::runtime_error("Model has no outputs.");
    }

    // Get output name
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();
    std::cout << "Output Name: " << output_name << std::endl;

    // 4. Prepare Input Data
    const size_t input_tensor_size = 1 * 10; // batch_size * num_features
    std::vector<float> input_tensor_values(input_tensor_size);
    // Fill the vector with a simple sequence: 0.0, 0.1, 0.2, ...
    for (size_t i = 0; i < input_tensor_size; ++i) {
        input_tensor_values[i] = static_cast<float>(i) * 0.1f;
    }
    print_vector(input_tensor_values, "\nInput Data: ");

    // 5. Create an ONNX Runtime Tensor object from our data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_dims.data(),
        input_dims.size()
    );

    // 6. Run Inference
    // We must use the C-style strings from our std::string objects for the names.
    std::vector<const char*> input_names = {input_name.c_str()};
    std::vector<const char*> output_names = {output_name.c_str()};

    std::cout << "\nRunning inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), &input_tensor, 1, // Array of inputs, number of inputs
        output_names.data(), 1                 // Array of outputs, number of outputs
    );
    std::cout << "Inference successful." << std::endl;

    // 7. Process the Output
    if (output_tensors.size() != 1 || !output_tensors.front().IsTensor()) {
        throw std::runtime_error("Failed to get a valid output tensor.");
    }

    // Get a pointer to the output data
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // Get the shape and size of the output tensor
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    size_t output_element_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    print_vector(output_shape, "Output Shape: ");
    std::cout << "Output Values (" << output_element_count << " elements):" << std::endl;
    for (size_t i = 0; i < output_element_count; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}