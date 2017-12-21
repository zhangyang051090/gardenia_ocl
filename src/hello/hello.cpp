#include <iostream>
#include <vector>
#include <vexcl/vexcl.hpp>

int main() {
    try {
        // Init VexCL context: grab one GPU with double precision.
        vex::Context ctx(
                vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
                vex::Filter::DoublePrecision &&
                vex::Filter::Count(1)
                );

        if (!ctx) throw std::runtime_error("GPUs with double precision not found");

        std::cout << ctx << std::endl;

        // Prepare input data.
        const size_t N = 1 << 20;

        std::vector<double> a(N, 1);
        std::vector<double> b(N, 2);
        std::vector<double> c(N);

        // Allocate device vectors and transfer input data to device.
        vex::vector<double> A(ctx.queue(), a);
        vex::vector<double> B(ctx.queue(), b);
        vex::vector<double> C(ctx.queue(), N);

        // Launch kernel on compute device.
        C = A + B;

        // Get result back to host.
        copy(C, c);
        
        // Should get '3' here.
        std::cout << c[42] << std::endl;
        return 0;
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error: " << err << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
    }
    return 1;
}
