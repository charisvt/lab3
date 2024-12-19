#include "xcl2.hpp"
#include "event_timer.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

#define lm 4
#define ln 4
#define lp 4

#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    EventTimer et;
    std::string binaryFile = argv[1];

    et.add("Allocate Memory in Host Memory");
    size_t total_elements_in1 = n * m;
    size_t total_elements_in2 = p * m;
    size_t total_elements_out = n * p;
    size_t vector1_size_bytes = sizeof(int) * total_elements_in1;
    size_t vector2_size_bytes = sizeof(int) * total_elements_in2;
    size_t vector0ut_size_bytes = sizeof(int) * total_elements_out;

    std::vector<unsigned int, aligned_allocator<unsigned int>> source_in1(total_elements_in1);
    std::vector<unsigned int, aligned_allocator<unsigned int>> source_in2(total_elements_in2);
    std::vector<unsigned int, aligned_allocator<unsigned int>> transposed_in2(total_elements_in2);
    std::vector<unsigned int, aligned_allocator<unsigned int>> source_hw_results(total_elements_out);
    std::vector<unsigned int> source_sw_results(total_elements_out);
    et.finish();

    et.add("Fill the buffers");
    for (unsigned int i = 0; i < total_elements_in1; i++) {
        source_in1[i] = static_cast<unsigned int>(std::rand() % 100);
    }
    for (unsigned i = 0; i < total_elements_in2; i++) {
        source_in2[i] = static_cast<unsigned int>(std::rand() % 100);
    }
    std::fill(source_hw_results.begin(), source_hw_results.end(), 0);
    std::fill(source_sw_results.begin(), source_sw_results.end(), 0);
    et.finish();

    et.add("Create transposed copy");
    // Create transposed copy of source_in2
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            transposed_in2[j * m + i] = source_in2[i * p + j];
        }
    }
    et.finish();

    et.add("Software Matrix Multiplication");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            unsigned int sum = 0;
            for (int k = 0; k < m; k++) {
                unsigned int a = source_in1[i * m + k];
                unsigned int b = transposed_in2[j * m + k]; // might as well use the transpose since we already calc'd it
                sum += a * b;
            }
            source_sw_results[i * p + j] = sum;
        }
    }
    et.finish();

    et.add("OpenCL host code");
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_matmul;
    cl::CommandQueue q;
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_matmul = cl::Kernel(program, "matmul", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cerr << "Failed to program any device found, exit!\n";
        return EXIT_FAILURE;
    }
    et.finish();

    et.add("Allocate Buffer in Global Memory");
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector1_size_bytes, source_in1.data(), &err));
    // notice we use the transposed version of in2
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector2_size_bytes, transposed_in2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector0ut_size_bytes, source_hw_results.data(), &err));
    et.finish();

    et.add("Set the Kernel Arguments");
    OCL_CHECK(err, err = krnl_matmul.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_matmul.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_matmul.setArg(2, buffer_out));
    OCL_CHECK(err, err = krnl_matmul.setArg(3, n));
    OCL_CHECK(err, err = krnl_matmul.setArg(4, m));
    OCL_CHECK(err, err = krnl_matmul.setArg(5, p));
    et.finish();

    et.add("Copy input data to device global memory");
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0));
    et.finish();

    et.add("Launch the Kernel");
    OCL_CHECK(err, q.enqueueTask(krnl_matmul));
    q.finish();
    et.finish();

    et.add("Copy Result from Device Global Memory to Host Local Memory");
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    et.finish();

    et.add("Validate Results");
    bool match = true;
    for (int i = 0; i < n * p; i++) {
        if (source_hw_results[i] != source_sw_results[i]) {
            std::cerr << "Mismatch at index " << i << ": HW=" << source_hw_results[i]
                      << ", SW=" << source_sw_results[i] << std::endl;
            match = false;
            //break;
        }
    }
    et.finish();

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    et.print();
    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}
