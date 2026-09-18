#pragma once

// Stable-API export macros.
//
// Scope: only headers under include/scicompute_attention/ are part of the public contract.
// Upper layers (vLLM(C++), RLHF, Python bindings) must include this directory and nothing else.

#if defined(_WIN32)
#define SCI_ATTENTION_API __declspec(dllexport)
#define SCI_ATTENTION_INTERNAL
#elif defined(__GNUC__) || defined(__clang__)
#define SCI_ATTENTION_API __attribute__((visibility("default")))
#define SCI_ATTENTION_INTERNAL __attribute__((visibility("hidden")))
#else
#define SCI_ATTENTION_API
#define SCI_ATTENTION_INTERNAL
#endif

// Public API convention:
//   * no exceptions cross the API boundary (every failure is a sci::Status);
//   * functions that produce a value return sci::Result<T>, empty returns use sci::Status;
//   * `stream == nullptr` means "use sci::Stream::GetCurrent()".

