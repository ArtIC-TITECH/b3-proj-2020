#include <torch/extension.h>
#include <cmath>

#define CHECK_TYPE(x,t) TORCH_CHECK(x.scalar_type() == t, "Type of" #x " must be " #t)
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel function for int8 quantizer / forward
void quantize_forward_cpu_kernel(const int64_t N, const float *input, float *output) {
    for (int64_t i = 0; i < N; i++) {
        float s = input[i];
        output[i] = (s < -128.0) ? -128.0 : ((s > 127.0) ? 127.0 : std::round(input[i]));
    }
}

// Wrapper function for int8 quantizer / forward
torch::Tensor quantize_forward (
    torch::Tensor input
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    CHECK_TYPE(input, at::ScalarType::Float);
    CHECK_CONTIGUOUS(input);

    quantize_forward_cpu_kernel(
        size,
        input.data_ptr<float>(),
        output.data_ptr<float>()
    );

    return output;
}

// Kernel function for int8 quantizer / backward
void quantize_backward_cpu_kernel(const int64_t N, const float *input, const float *grad_out, float *grad_in) {
    for (int64_t i = 0; i < N; i++) {
        float s = input[i];
        grad_in[i] = static_cast<float>((s >= -128.0) & (s <= 127.0)) * grad_out[i];
    }
}

// Wrapper function for int8 quantizer / backward
torch::Tensor quantize_backward (
    torch::Tensor input,      // u
    torch::Tensor grad_out    // dL/dy
) {
    auto size = input.numel();
    auto grad_in = torch::empty_like(input);
    TORCH_CHECK(grad_out.numel() == size, "Shape mismatch");
    CHECK_TYPE(input, at::ScalarType::Float);
    CHECK_CONTIGUOUS(grad_out);

    quantize_backward_cpu_kernel(
        size,
        input.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_in.data_ptr<float>()
    );

    return grad_in;  // dL/du
}


// Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_forward", &quantize_forward, "Emulates int8 quantization in float.",
        py::arg("input")
    );
    m.def("quantize_backward", &quantize_backward, "Backward function for int8 quantization.",
        py::arg("input"),
        py::arg("grad_out")
    );
}