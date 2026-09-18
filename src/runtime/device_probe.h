// C ABI between host-only translation units and the CUDA probe TU.
//
// Rationale: src/api/* and most of src/runtime/* must stay CPU-compilable (they are also parsed
// when the project is consumed by a C++-only toolchain), so all cudaXxx calls are confined to
// .cu files behind this small C interface.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define SCA_DEVICE_NAME_MAX 256

typedef struct ScaDeviceProbe {
    char name[SCA_DEVICE_NAME_MAX];
    int major;
    int minor;
    int sm_count;
    unsigned long long smem_per_sm;
    unsigned long long smem_per_block_default;
    unsigned long long smem_per_block_optin;
    unsigned long long regs_per_sm;
    unsigned long long max_threads_per_sm;
    unsigned long long l2_bytes;
    unsigned long long total_mem_bytes;
    unsigned long long free_mem_bytes;
    int cuda_available;
    int binary_has_sm80;  // this binary contains >= sm_80 code (mma.sync / cp.async)
    int binary_has_sm90;  // this binary contains >= sm_90 code (TMA bulk)
} ScaDeviceProbe;

// Fills `out` for the given device. Returns 0 on success, non-zero CUDA error code otherwise.
int sca_probe_device(int device_id, ScaDeviceProbe* out);

// cudaMemGetInfo wrapper. Returns 0 on success.
int sca_mem_get_info(unsigned long long* free_bytes, unsigned long long* total_bytes);

#ifdef __cplusplus
}
#endif

