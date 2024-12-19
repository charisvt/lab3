#include <ap_int.h>
#include <stdio.h>
#include <string.h>

#define DATAWIDTH 512
#define VECTOR_SIZE (DATAWIDTH / 32) // vector size is 16 (512/32 = 16)
typedef ap_uint<DATAWIDTH> uint512_dt;
typedef ap_uint<32> uint32;

/*
    Matrix Multiplication Kernel Implementation using uint512_dt datatype
    Arguments:
        in1   (input)     --> Input Matrix 1 with dimensions n x m
        in2   (input)     --> Transposed Input Matrix 2 with dimensions p x m
        out   (output)    --> Output Matrix with dimensions n x p
        n, m, p           --> Dimensions of matrices
*/

extern "C" {
void matmul(
		const uint512_dt *in1,
		const uint512_dt *in2,
		uint512_dt *out,
		int n, int m, int p
		){

#pragma HLS INTERFACE m_axi bundle=gmem port = in1
#pragma HLS INTERFACE m_axi bundle=gmem1 port = in2
#pragma HLS INTERFACE m_axi bundle=gmem2 port = out
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = m bundle = control
#pragma HLS INTERFACE s_axilite port = p bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
// we use different memory banks to maximize bandwidth

	// double buffers for in1 and in2
    uint512_dt buf_in1_a, buf_in1_b;
    uint512_dt buf_in2_a, buf_in2_b;

	// preload in1 buffer (single preload)
	buf_in1_a = in1[0];

outer:
	for(int i = 0; i < n; i++){
#pragma HLS LOOP_TRIPCOUNT min = 1 max = n
#pragma HLS PIPELINE II=1
		// start preloading next vector from in1
        if (i + 1 < n) buf_in1_b = in1[i + 1];

        // start preloading in2 buffer
        buf_in2_a = in2[0];
	    uint512_dt out_vec = 0;

	middle:
	    for(int j = 0; j < p; j++){
#pragma HLS LOOP_TRIPCOUNT min = 1 max = p
#pragma HLS PIPELINE II=1
	    	// start preloading next vector from in2
	    	if(j + 1 < p) buf_in2_b = in2[j + 1];

	    	uint32 sum = 0;
            uint512_dt temp_in1 = buf_in1_a;
            uint512_dt temp_in2 = buf_in2_a;

        inner:
            for(int k = 0; k < VECTOR_SIZE; k++){
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = 1 max = VECTOR_SIZE
                // extract elements from vector
                uint32 a = temp_in1.range(32 * (k + 1) -1, 32 * k);
                uint32 b = temp_in2.range(32 * (k + 1) -1, 32 * k);
                sum += a * b;
            }
            uint32 elem_idx = j % VECTOR_SIZE;
            out_vec.range(32 * elem_idx + 31, 32 * elem_idx) = sum;

            // swap buffers for in2
            buf_in2_a = buf_in2_b;
	    }
        out[i] = out_vec;
        // swap buffers for in1
        buf_in1_a = buf_in1_b;
	}
}
}

