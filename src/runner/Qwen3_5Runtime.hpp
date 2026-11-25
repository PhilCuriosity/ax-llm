#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace axllm::qwen3_5 {

// Configuration and state policy for Qwen3.5's mixed linear/full attention.
// Device runners stay in the existing LLM implementation; this class owns only
// model policy and opaque recurrent-state metadata so it is backend agnostic.
struct AttentionConfig {
    std::vector<std::string> layer_types;
    int full_attention_interval = 0;
    int num_kv_shared_layers = 0;
    int sliding_window = 0;
};

// One device-side allocation. ``owner`` keeps the CMM buffer alive and frees
// it through the backend-specific deleter when the snapshot is dropped.
struct DeviceBuffer {
    void *vir = nullptr;
    unsigned long long phy = 0;
    std::size_t size = 0;
    std::shared_ptr<void> owner;
};

// Linear-attention tensors are opaque state buffers. Keep raw bytes because
// exported graphs may use BF16/FP16 or FP32 depending on the decode path.
// When ``device_backed`` is true the K/V state lives in CMM and k/v are empty;
// otherwise k/v hold the original host-side copies (AXCL/host fallback).
struct LinearStateSnapshot {
    int token_len = 0;
    bool device_backed = false;
    std::vector<std::vector<unsigned char>> k;
    std::vector<std::vector<unsigned char>> v;
    std::vector<DeviceBuffer> device_k;
    std::vector<DeviceBuffer> device_v;
};

class LinearStateStore {
public:
    void clear() { snapshots_.clear(); }
    bool empty() const { return snapshots_.empty(); }
    void drop_after(int token_len);
    int best_len(int token_len) const;
    LinearStateSnapshot *find(int token_len);
    const LinearStateSnapshot *find(int token_len) const;

    std::vector<LinearStateSnapshot> &items() { return snapshots_; }
    const std::vector<LinearStateSnapshot> &items() const { return snapshots_; }

private:
    std::vector<LinearStateSnapshot> snapshots_;
};

class Runtime {
public:
    Runtime() = default;
    Runtime(int layer_count, AttentionConfig config);

    void configure(int layer_count, AttentionConfig config);

    bool is_linear_layer(int layer_idx) const;
    bool has_linear_attention_layers() const;
    bool is_sliding_attention_layer(int layer_idx) const;
    int first_full_layer_idx() const;
    int shared_kv_source_for_layer(int layer_idx) const;

    // Return the uncached suffix and its common-prefix length.
    static std::vector<int> token_suffix(const std::vector<int> &cached,
                                         const std::vector<int> &requested,
                                         int &common_prefix);

    static size_t cache_bytes(uint32_t n_size);
    static size_t cache_u16_slots(uint32_t n_size);
    static bool compatible_cache_dtype(int src_dtype, int dst_dtype);

private:
    std::vector<bool> linear_layers_;
    std::vector<bool> sliding_layers_;
    std::vector<int> shared_kv_source_layers_;
    int sliding_window_ = 0;
};

} // namespace axllm::qwen3_5
