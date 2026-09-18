// SciCompute-Attention architecture probe.
//
// Purpose : verify, on the *actual* device, every instruction-level capability the attention
//           kernels depend on. The prompt forbids designing kernels from memory, so every
//           probe below is a runtime check (compile + execute + value verification).
// Usage   : nvcc -std=c++17 -O2 -arch=sm_XXX arch_probe.cu -lcuda -o arch_probe && ./arch_probe [--json]
//
// Probes
//   [1] mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  -> identity-GEMM check (C == B)
//   [2] cp.async.cg.shared.global 16B + commit_group/wait_group
//   [3] ldmatrix.sync.aligned.m8n8.x4.shared.b16           -> 16x16 tile fragment reconstruction
//   [4] cp.async.bulk.shared::cta.global (TMA 1D) with mbarrier
//   [5] cp.async.bulk.tensor.2d.shared::cta.global (TMA tensor 2D, host-built CUtensorMap)
//
// wgmma / tcgen05 are *not* probed here: a translation unit that must compile cannot contain
// them (ptxas rejects them on sm_120). They are covered by the expected-to-fail translation
// unit tools/arch_probe/wgmma_probe.cu, driven by run_probe.sh.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (expr);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(err__));                                         \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

// ---------------------------------------------------------------------------------------
// device-side half packing helpers (std::memcpy is not device-callable)
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t PackHalf2(__half lo, __half hi) {
    const __half2 h2 = __halves2half2(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ __half LoHalf(uint32_t packed) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&packed);
    return __low2half(h2);
}

__device__ __forceinline__ __half HiHalf(uint32_t packed) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&packed);
    return __high2half(h2);
}

// ---------------------------------------------------------------------------------------
// [1] mma.sync m16n8k16 f16*f16+f32
//
// math: C[m][n] = sum_k A[m][k] * B[k][n]; with A = I this yields C == B, which validates the
//       fragment layout the flash kernel relies on (PTX ISA 9.7.14.5.10):
//   A (16x16, 4 x b32 = 8 halves):
//     a0,a1 = A[g][2t+0..1]   a2,a3 = A[g+8][2t+0..1]
//     a4,a5 = A[g][2t+8..9]   a6,a7 = A[g+8][2t+8..9]
//   B (16x8, 2 x b32 = 4 halves), fed here as B[k][n] row-major:
//     b0,b1 = B[2t+0..1][g]   b2,b3 = B[2t+8..9][g]
//   C (16x8, 4 x f32):
//     c0 = C[g][2t]  c1 = C[g][2t+1]  c2 = C[g+8][2t]  c3 = C[g+8][2t+1]
//   with g = lane >> 2, t = lane & 3.
// ---------------------------------------------------------------------------------------
__global__ void probe_mma_m16n8k16(const float* b_matrix, int* ok_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int a_rows[8] = {g, g, g + 8, g + 8, g, g, g + 8, g + 8};
    const int a_cols[8] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8,
                           2 * t + 9};
    uint32_t a[4];
    for (int r = 0; r < 4; ++r) {
        const __half lo = __float2half(a_rows[2 * r] == a_cols[2 * r] ? 1.0f : 0.0f);
        const __half hi = __float2half(a_rows[2 * r + 1] == a_cols[2 * r + 1] ? 1.0f : 0.0f);
        a[r] = PackHalf2(lo, hi);
    }

    const int b_rows[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
    uint32_t b[2];
    for (int r = 0; r < 2; ++r) {
        b[r] = PackHalf2(__float2half(b_matrix[b_rows[2 * r] * 8 + g]),
                         __float2half(b_matrix[b_rows[2 * r + 1] * 8 + g]));
    }

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

    const int c_rows[4] = {g, g, g + 8, g + 8};
    const int c_cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        const float expect = b_matrix[c_rows[i] * 8 + c_cols[i]];
        if (fabsf(c[i] - expect) > 1e-5f) ok = false;
    }
    if (__syncthreads_or(!ok) && threadIdx.x == 0) atomicAdd(ok_flag, 1);
#else
    (void)b_matrix;
    (void)ok_flag;
#endif
}

// ---------------------------------------------------------------------------------------
// [2] cp.async.cg.shared.global 16B + commit_group / wait_group
// ---------------------------------------------------------------------------------------
__global__ void probe_cp_async(const float* src, float* dst, int n, int* ok_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    for (int i = tid; i < n / 4; i += blockDim.x) {
        const uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[i * 4]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s_addr),
                     "l"(reinterpret_cast<const char*>(src) + i * 16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    bool ok = true;
    for (int i = tid; i < n; i += blockDim.x) {
        if (smem[i] != src[i]) ok = false;
        dst[i] = smem[i];
    }
    if (__syncthreads_or(!ok) && tid == 0) atomicAdd(ok_flag, 1);
#else
    (void)src;
    (void)dst;
    (void)n;
    (void)ok_flag;
#endif
}

// ---------------------------------------------------------------------------------------
// [3] ldmatrix.sync.aligned.m8n8.x4.shared.b16
//
// Source tile: 16x16 halves, row-major, row stride 16 halves.
// Address supply (PTX ISA 9.7.14.4.15): lanes 0-7 -> matrix0 (rows 0-7, cols 0-7),
// 8-15 -> matrix1 (rows 8-15, cols 0-7), 16-23 -> matrix2 (rows 0-7, cols 8-15),
// 24-31 -> matrix3 (rows 8-15, cols 8-15).
// Result: register rm holds matrix m; lane receives row (lane>>2), columns 2*(lane&3) and +1
// relative to that matrix' column origin.
// ---------------------------------------------------------------------------------------
__global__ void probe_ldmatrix(const __half* src, __half* frag_out, int* ok_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    __shared__ __half tile[16 * 16];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) tile[i] = src[i];
    __syncthreads();

    const int lane = tid & 31;
    const int m = lane >> 3;  // which 8x8 matrix this lane supplies an address for
    const int r = lane & 7;   // row inside that matrix
    const int row_in_tile = ((m & 1) ? 8 : 0) + r;
    const int col_origin = ((m & 2) ? 8 : 0);
    const uint32_t addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&tile[row_in_tile * 16 + col_origin]));

    uint32_t regs[4] = {0, 0, 0, 0};
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));

    bool ok = true;
    for (int rm = 0; rm < 4; ++rm) {
        const __half lo = LoHalf(regs[rm]);
        const __half hi = HiHalf(regs[rm]);
        const int src_row = ((rm & 1) ? 8 : 0) + (lane >> 2);
        const int src_col = ((rm & 2) ? 8 : 0) + 2 * (lane & 3);
        const __half expect_lo = src[src_row * 16 + src_col];
        const __half expect_hi = src[src_row * 16 + src_col + 1];
        if (__half2float(lo) != __half2float(expect_lo)) ok = false;
        if (__half2float(hi) != __half2float(expect_hi)) ok = false;
        frag_out[tid * 8 + 2 * rm + 0] = lo;
        frag_out[tid * 8 + 2 * rm + 1] = hi;
    }
    if (__syncthreads_or(!ok) && tid == 0) atomicAdd(ok_flag, 1);
#else
    (void)src;
    (void)frag_out;
    (void)ok_flag;
#endif
}

// ---------------------------------------------------------------------------------------
// [4] cp.async.bulk (TMA 1D) + mbarrier completion
// ---------------------------------------------------------------------------------------
__global__ void probe_tma_bulk(const float* src, float* dst, int bytes, int* ok_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) unsigned char smem_bulk_raw[];
    __shared__ __align__(8) uint64_t bar;
    const int tid = threadIdx.x;
    const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(bar_addr));
        asm volatile("fence.proxy.async.shared::cta;\n" ::);
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, "
            "[%3];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_bulk_raw))),
            "l"(src), "r"(bytes), "r"(bar_addr));
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(bar_addr),
                     "r"(bytes));
    }
    __syncthreads();
    int done = 0;
    while (!done) {
        asm volatile(
            "{.reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
            " selp.u32 %0, 1, 0, p;}\n"
            : "=r"(done)
            : "r"(bar_addr));
    }
    __syncthreads();

    const float* smem_f = reinterpret_cast<const float*>(smem_bulk_raw);
    const int n = bytes / 4;
    bool ok = true;
    for (int i = tid; i < n; i += blockDim.x) {
        if (smem_f[i] != src[i]) ok = false;
        dst[i] = smem_f[i];
    }
    if (__syncthreads_or(!ok) && tid == 0) atomicAdd(ok_flag, 1);
#else
    (void)src;
    (void)dst;
    (void)bytes;
    (void)ok_flag;
#endif
}

// ---------------------------------------------------------------------------------------
// [5] cp.async.bulk.tensor.2d (TMA tensor 2D) with a host-built CUtensorMap
// ---------------------------------------------------------------------------------------
__global__ void probe_tma_tensor2d(const __grid_constant__ CUtensorMap tmap, float* dst,
                                   int rows, int cols, int* ok_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) unsigned char smem_tensor_raw[];
    __shared__ __align__(8) uint64_t bar;
    const int tid = threadIdx.x;
    const int bytes = rows * cols * static_cast<int>(sizeof(float));
    const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(bar_addr));
        asm volatile("fence.proxy.async.shared::cta;\n" ::);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(
                static_cast<uint32_t>(__cvta_generic_to_shared(smem_tensor_raw))),
            "l"(reinterpret_cast<uint64_t>(&tmap)), "r"(0), "r"(0), "r"(bar_addr));
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(bar_addr),
                     "r"(bytes));
    }
    __syncthreads();
    int done = 0;
    while (!done) {
        asm volatile(
            "{.reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
            " selp.u32 %0, 1, 0, p;}\n"
            : "=r"(done)
            : "r"(bar_addr));
    }
    __syncthreads();
    const float* smem_f = reinterpret_cast<const float*>(smem_tensor_raw);
    bool ok = true;
    for (int i = tid; i < rows * cols; i += blockDim.x) {
        dst[i] = smem_f[i];
        if (smem_f[i] != static_cast<float>(i)) ok = false;
    }
    if (__syncthreads_or(!ok) && tid == 0) atomicAdd(ok_flag, 1);
#else
    (void)tmap;
    (void)dst;
    (void)rows;
    (void)cols;
    (void)ok_flag;
#endif
}

struct ProbeResult {
    std::string name;
    std::string status;  // PASS / FAIL / SKIP
    std::string detail;
};

static void PrintProps() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock = 0, l2 = 0;
    cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    std::printf("[props] name=%s cc=%d.%d sm=%d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("[props] smem/SM=%zu smem/block(default)=%zu smem/block(optin)=%zu\n",
                prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock,
                prop.sharedMemPerBlockOptin);
    std::printf("[props] regs/SM=%d threads/SM=%d threads/block=%d warp=%d L2=%d\n",
                prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor,
                prop.maxThreadsPerBlock, prop.warpSize, l2);
    std::printf("[props] mem.total=%zu mem.clock=%d kHz bus.width=%d\n", prop.totalGlobalMem,
                mem_clock, prop.memoryBusWidth);
}

int main(int argc, char** argv) {
    const bool json = (argc > 1 && std::string(argv[1]) == "--json");
    std::vector<ProbeResult> results;
    PrintProps();

    int* d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    auto read_flag = [&]() {
        int flag = 0;
        CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        return flag;
    };

    // ---- [1] mma.sync m16n8k16 ---------------------------------------------------------
    {
        float h_b[16 * 8];
        for (int k = 0; k < 16; ++k)
            for (int n = 0; n < 8; ++n) h_b[k * 8 + n] = static_cast<float>(k * 8 + n) * 0.25f;
        float* d_b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_b, sizeof(h_b)));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));
        probe_mma_m16n8k16<<<1, 32>>>(d_b, d_flag);
        CUDA_CHECK(cudaDeviceSynchronize());
        const int flag = read_flag();
        results.push_back({"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
                           flag == 0 ? "PASS" : "FAIL",
                           flag == 0 ? "identity-GEMM C==B verified over 32 lanes"
                                     : "fragment layout mismatch"});
        cudaFree(d_b);
    }

    // ---- [2] cp.async.cg ---------------------------------------------------------------
    {
        const int n = 1024;
        std::vector<float> h_src(n), h_dst(n, 0.f);
        for (int i = 0; i < n; ++i) h_src[i] = static_cast<float>(i) + 0.5f;
        float *d_src = nullptr, *d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        probe_cp_async<<<1, 128, n * sizeof(float)>>>(d_src, d_dst, n, d_flag);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));
        const bool ok = (read_flag() == 0) &&
                        (std::memcmp(h_src.data(), h_dst.data(), n * sizeof(float)) == 0);
        results.push_back({"cp.async.cg.shared.global + commit_group/wait_group",
                           ok ? "PASS" : "FAIL",
                           ok ? "16B async copy of 4 KiB verified" : "async copy mismatch"});
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    // ---- [3] ldmatrix.x4 ---------------------------------------------------------------
    {
        std::vector<__half> h_src(256);
        for (int i = 0; i < 256; ++i) h_src[i] = __float2half(static_cast<float>(i));
        __half* d_src = nullptr;
        __half* d_frag = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, h_src.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_frag, 32 * 8 * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        probe_ldmatrix<<<1, 32>>>(d_src, d_frag, d_flag);
        CUDA_CHECK(cudaDeviceSynchronize());
        const int flag = read_flag();
        results.push_back({"ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                           flag == 0 ? "PASS" : "FAIL",
                           flag == 0 ? "16x16 tile fragment layout verified"
                                     : "fragment layout mismatch"});
        cudaFree(d_src);
        cudaFree(d_frag);
    }

    // ---- [4] TMA 1D bulk ---------------------------------------------------------------
    {
        const int n = 512;
        std::vector<float> h_src(n), h_dst(n, 0.f);
        for (int i = 0; i < n; ++i) h_src[i] = static_cast<float>(i) * 2.f;
        float *d_src = nullptr, *d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        probe_tma_bulk<<<1, 128, n * sizeof(float)>>>(d_src, d_dst, n * sizeof(float), d_flag);
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            results.push_back({"cp.async.bulk.shared::cta.global (TMA 1D)", "FAIL",
                               std::string("runtime: ") + cudaGetErrorString(err)});
            cudaGetLastError();
        } else {
            CUDA_CHECK(
                cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));
            const bool ok = (read_flag() == 0) &&
                            (std::memcmp(h_src.data(), h_dst.data(), n * sizeof(float)) == 0);
            results.push_back({"cp.async.bulk.shared::cta.global (TMA 1D)",
                               ok ? "PASS" : "FAIL",
                               ok ? "2 KiB bulk copy with mbarrier verified"
                                  : "bulk copy mismatch"});
        }
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    // ---- [5] TMA tensor 2D -------------------------------------------------------------
    {
        const int rows = 32, cols = 32;
        std::vector<float> h_src(rows * cols), h_dst(rows * cols, 0.f);
        for (int i = 0; i < rows * cols; ++i) h_src[i] = static_cast<float>(i);
        float *d_src = nullptr, *d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, h_src.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, h_src.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUtensorMap tmap{};
        cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(cols),
                                    static_cast<cuuint64_t>(rows)};
        cuuint64_t global_stride[1] = {static_cast<cuuint64_t>(cols * sizeof(float))};
        cuuint32_t box_dim[2] = {static_cast<cuuint32_t>(cols), static_cast<cuuint32_t>(rows)};
        cuuint32_t elem_stride[2] = {1, 1};
        const CUresult res = cuTensorMapEncodeTiled(
            &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, d_src, global_dim, global_stride,
            box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            results.push_back({"cp.async.bulk.tensor.2d (TMA tensor 2D)", "SKIP",
                               "cuTensorMapEncodeTiled unavailable"});
        } else {
            const int smem_bytes = static_cast<int>(h_src.size() * sizeof(float));
            probe_tma_tensor2d<<<1, 128, smem_bytes>>>(tmap, d_dst, rows, cols, d_flag);
            const cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                results.push_back({"cp.async.bulk.tensor.2d (TMA tensor 2D)", "FAIL",
                                   std::string("runtime: ") + cudaGetErrorString(err)});
                cudaGetLastError();
            } else {
                CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, h_src.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                const bool ok = std::memcmp(h_src.data(), h_dst.data(),
                                            h_src.size() * sizeof(float)) == 0;
                results.push_back({"cp.async.bulk.tensor.2d (TMA tensor 2D)",
                                   ok ? "PASS" : "FAIL",
                                   ok ? "32x32 fp32 tile via CUtensorMap verified"
                                      : "tensor tile mismatch"});
            }
        }
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    cudaFree(d_flag);

    if (json) {
        std::printf("{\n  \"probes\": [\n");
        for (size_t i = 0; i < results.size(); ++i) {
            std::printf("    {\"name\": \"%s\", \"status\": \"%s\", \"detail\": \"%s\"}%s\n",
                        results[i].name.c_str(), results[i].status.c_str(),
                        results[i].detail.c_str(), i + 1 == results.size() ? "" : ",");
        }
        std::printf("  ]\n}\n");
    } else {
        std::printf("\n");
        for (const auto& r : results) {
            std::printf("[probe] %-56s : %-4s  (%s)\n", r.name.c_str(), r.status.c_str(),
                        r.detail.c_str());
        }
    }

    for (const auto& r : results) {
        if (r.status == "FAIL") return 1;
    }
    return 0;
}
