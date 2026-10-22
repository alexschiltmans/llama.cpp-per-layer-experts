#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "llama-moe-cache-io.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

bool llama_moe_cache_direct_io() {
    const char * v = getenv("LLAMA_MOE_CACHE_DIRECT_IO");
    return v != nullptr && atoi(v) != 0;
}

// LLAMA_MOE_CACHE_DIRECT_IO_THREADS=1..32: the number of reads in flight, 1 reads one run of experts after the other
static int llama_moe_cache_direct_io_threads() {
    const char * v = getenv("LLAMA_MOE_CACHE_DIRECT_IO_THREADS");
    if (v == nullptr || *v == '\0') {
        return 8;
    }
    char * end = nullptr;
    const long n = strtol(v, &end, 10);
    if (*end != '\0' || n < 1 || n > 32) {
        throw std::runtime_error(format("LLAMA_MOE_CACHE_DIRECT_IO_THREADS must be an integer from 1 to 32, not '%s'", v));
    }
    return (int) n;
}

// LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR=<index>: direct I/O reads the missing experts from an expert-major sidecar of the model file,
// whose index is written by the sidecar builder, nullptr if not set
static const char * llama_moe_cache_direct_io_sidecar() {
    const char * v = getenv("LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR");
    return v != nullptr && *v != '\0' ? v : nullptr;
}

namespace {

// a model file opened for reads that bypass the page cache
struct moe_cache_file {
    int    fd   = -1;
    size_t size = 0;
};

#if defined(__APPLE__) || defined(__linux__)
static size_t moe_cache_page_size() {
#if defined(__APPLE__)
    return (size_t) sysconf(_SC_PAGESIZE);
#else
    // O_DIRECT needs the buffer, offset and length aligned to the logical block size, at most 4 KiB on common disks
    return std::max((size_t) sysconf(_SC_PAGESIZE), (size_t) 4096);
#endif
}

#if defined(__APPLE__)
static moe_cache_file moe_cache_file_open(const std::string & path) {
    moe_cache_file f;
    f.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (f.fd < 0 || fcntl(f.fd, F_NOCACHE, 1) == -1 || fstat(f.fd, &st) != 0) {
        const std::string err = strerror(errno);
        if (f.fd >= 0) {
            close(f.fd);
        }
        throw std::runtime_error(format("MoE cache direct I/O: cannot open %s without the page cache: %s", path.c_str(), err.c_str()));
    }
    f.size = (size_t) st.st_size;
    return f;
}
#else
// O_DIRECT; some filesystems accept it at open and fail the reads, so one aligned page is read here
static moe_cache_file moe_cache_file_open(const std::string & path) {
    moe_cache_file f;
    std::string err;
    f.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    struct stat st;
    if (f.fd < 0 || fstat(f.fd, &st) != 0) {
        err = strerror(errno);
    } else {
        f.size = (size_t) st.st_size;
        const size_t page = moe_cache_page_size();
        void * buf = nullptr;
        if (posix_memalign(&buf, page, page) != 0) {
            err = "out of memory";
        } else {
            ssize_t n;
            do {
                n = pread(f.fd, buf, page, 0);
            } while (n < 0 && errno == EINTR);
            if (n < 0) {
                err = strerror(errno);
            } else if ((size_t) n != std::min(page, f.size)) {
                err = "short read";
            }
            free(buf);
        }
    }
    if (!err.empty()) {
        if (f.fd >= 0) {
            close(f.fd);
        }
        throw std::runtime_error(format("MoE cache direct I/O: cannot read %s with O_DIRECT, without the page cache "
            "(it needs a local filesystem with O_DIRECT, not a WSL /mnt drive, a network or a FUSE mount): %s", path.c_str(), err.c_str()));
    }
    return f;
}
#endif

static void moe_cache_file_close(moe_cache_file & f) {
    if (f.fd >= 0) {
        close(f.fd);
        f.fd = -1;
    }
}

// CLOCK_MONOTONIC in microseconds, the clock of ggml_time_us, for matching the trace with outside samples
static int64_t moe_cache_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec*1000000 + ts.tv_nsec/1000;
}

// the page-aligned range of a model file read for the bytes [first, first + n) of it: size bytes from begin,
// of which the first need bytes must be read
struct moe_cache_span {
    size_t begin;
    size_t size;
    size_t need;
};

static moe_cache_span moe_cache_file_span(const moe_cache_file & f, size_t first, size_t n, size_t page_size) {
    const size_t begin = first & ~(page_size - 1);
#if defined(__APPLE__)
    // F_NOCACHE reads need no alignment: the last page is cut at the end of the file
    const size_t end = std::min((size_t) GGML_PAD(first + n, page_size), f.size);
    return { begin, end - begin, end - begin };
#else
    // O_DIRECT reads whole pages: a read of the last page stops short at the end of the file, after the needed bytes
    const size_t end = GGML_PAD(first + n, page_size);
    return { begin, end - begin, std::min(end, f.size) - begin };
#endif
}

// read size bytes at offs, abort on an I/O error or an end of file before need bytes
// (need < size only at the end of the file with O_DIRECT, which stops short there)
static void moe_cache_file_read(const moe_cache_file & f, uint8_t * dst, size_t size, size_t offs, size_t need) {
    while (size > 0) {
        const ssize_t n = pread(f.fd, dst, std::min(size, (size_t) 1 << 30), (off_t) offs);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            GGML_ABORT("MoE cache direct I/O: read of %zu bytes at offset %zu failed: %s", size, offs, n == 0 ? "end of file" : strerror(errno));
        }
        dst  += n;
        size -= n;
        offs += n;
        if ((size_t) n >= need) {
            return;
        }
        need -= n;
    }
}
#else
struct moe_cache_span {
    size_t begin;
    size_t size;
    size_t need;
};
static moe_cache_file moe_cache_file_open(const std::string & path) {
    throw std::runtime_error(format("MoE cache direct I/O is only implemented on macOS and Linux (%s)", path.c_str()));
}
static void moe_cache_file_close(moe_cache_file &) {}
static size_t moe_cache_page_size() { return 4096; }
static int64_t moe_cache_time_us() { return 0; }
static moe_cache_span moe_cache_file_span(const moe_cache_file &, size_t, size_t, size_t) { GGML_ABORT("MoE cache direct I/O is only implemented on macOS and Linux"); }
static void moe_cache_file_read(const moe_cache_file &, uint8_t *, size_t, size_t, size_t) { GGML_ABORT("MoE cache direct I/O is only implemented on macOS and Linux"); }
#endif

#if defined(__APPLE__) || defined(__linux__)
using moe_cache_read      = llama_moe_cache_read;
using moe_cache_read_pool = llama_moe_cache_read_pool;
#else
struct moe_cache_read { int fd; uint8_t * dst; size_t size; size_t offs; size_t need; };
struct moe_cache_read_pool {
    explicit moe_cache_read_pool(int) {}
    void run(const moe_cache_read *, size_t, int *) { GGML_ABORT("MoE cache direct I/O is only implemented on macOS and Linux"); }
};
#endif

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    using fill = llama_moe_cache_fill;

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = llama_moe_cache::max_batch;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // direct I/O: the missing experts are read from the model files into a page-aligned staging buffer
    struct source {
        int32_t file;
        size_t  offs;
    };
    bool direct_io = false;
    std::vector<moe_cache_file> files;                         // [n_files] of the model, opened on first use
    std::unordered_map<const ggml_tensor *, source> sources;   // host experts -> file and offset of their data
    std::vector<uint8_t> staging;
    size_t page_size  = 0;
    size_t dio_reads  = 0;
    size_t dio_bytes  = 0; // read from the files, including the page alignment, up to the end of the file
    int32_t il_trace  = -1; // LLAMA_MOE_CACHE_DIRECT_IO_TRACE=1: log the bytes read after each ubatch, at this layer

    // with more than one read thread, the runs of missing experts of a layer are read in parallel before they are uploaded
    struct upload {
        ggml_tensor * bank;
        size_t bank_offs;
        size_t size;
        size_t staging_offs; // of the read, in the page-aligned staging buffer
        size_t skip;         // from the start of the read to the first expert
    };
    static constexpr size_t max_staging = (size_t) 256 << 20; // per wave of parallel reads, a larger read is a wave of its own
    int n_io_threads = 1;
    std::unique_ptr<moe_cache_read_pool> io_pool;
    std::vector<moe_cache_read> io_reads;
    std::vector<upload> io_uploads;
    std::vector<int> io_res;

    // LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR: the missing experts are read from an expert-major sidecar, one read per run of consecutive experts
    bool sidecar = false;
    std::string sidecar_index;                              // path of the index
    moe_cache_file sidecar_file;
    std::vector<llama_moe_cache_sidecar_layout> sidecar_layouts; // [n_layer], the parts in the order of the banks
    std::vector<llama_moe_cache_sidecar_job> sc_jobs;
    std::vector<llama_moe_cache_sidecar_upload> sc_uploads;
    std::vector<llama_moe_cache_sidecar_wave> sc_waves;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (llama_moe_cache_direct_io_sidecar() != nullptr && !llama_moe_cache_direct_io()) {
            throw std::runtime_error("LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR needs LLAMA_MOE_CACHE_DIRECT_IO=1");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            // with direct I/O no layer may compute from the mapped experts
            const bool any_host = std::any_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                if (any_host && llama_moe_cache_direct_io()) {
                    throw std::runtime_error(format("MoE cache direct I/O: layer %zu keeps only some of its experts in host memory, they cannot be cached", il));
                }
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                if (llama_moe_cache_direct_io()) {
                    throw std::runtime_error(format("MoE cache direct I/O: layer %zu is not assigned to a GPU, its experts cannot be cached", il));
                }
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                if (llama_moe_cache_direct_io()) {
                    throw std::runtime_error(format("MoE cache direct I/O: the budget is too small for %zu layers, their experts would be read from the mapped weights", g.layers.size()));
                }
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (llama_moe_cache_direct_io() && !model.hparams.no_alloc) {
            init_direct_io(model);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        for (moe_cache_file & f : files) {
            moe_cache_file_close(f);
        }
        moe_cache_file_close(sidecar_file);
    }

    // every cached expert tensor must be the data of a model file at a known offset, the files are opened here
    void init_direct_io(const llama_model & model) {
        direct_io = true;
        page_size = moe_cache_page_size();
        files.resize(model.source_paths.size());
        for (const layer & l : layers) {
            for (const ggml_tensor * w : l.experts) {
                const auto it = model.weight_sources.find(w);
                if (it == model.weight_sources.end()) {
                    throw std::runtime_error(format("MoE cache direct I/O: the data of %s is not mapped from a model file (load with mmap and without repacking)", w->name));
                }
                const int32_t file = (int32_t) it->second.file;
                if (files[file].fd < 0) {
                    files[file] = moe_cache_file_open(model.source_paths[file]);
                }
                if (it->second.offs + ggml_nbytes(w) > files[file].size) {
                    throw std::runtime_error(format("MoE cache direct I/O: %s is outside of %s", w->name, model.source_paths[file].c_str()));
                }
                sources[w] = { file, it->second.offs };
            }
        }
        n_io_threads = llama_moe_cache_direct_io_threads();
        const char * sidecar_env = llama_moe_cache_direct_io_sidecar();
        if (sidecar_env != nullptr) {
            init_sidecar(model, sidecar_env);
        }
        // the sidecar reads always go through the pool, with one thread the calling thread reads them one after the other
        if (n_io_threads > 1 || sidecar) {
            io_pool = std::make_unique<moe_cache_read_pool>(n_io_threads);
        }
        LLAMA_LOG_INFO("%s: MoE cache direct I/O: the missing experts of %zu tensors are read from the model files without the page cache, %d read threads%s%s\n", __func__, sources.size(), n_io_threads,
            sidecar ? ", sidecar " : "", sidecar ? sidecar_index.c_str() : "");
        const char * trace = getenv("LLAMA_MOE_CACHE_DIRECT_IO_TRACE");
        if (trace != nullptr && atoi(trace) != 0) {
            for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
                if (layers[il].ig >= 0) {
                    il_trace = il;
                }
            }
        }
    }

    // size bytes of f at offs, read with page-aligned bounds into buf; not counted in the direct I/O stats
    const uint8_t * read_check(const moe_cache_file & f, size_t offs, size_t size, std::vector<uint8_t> & buf) const {
        const moe_cache_span sp = moe_cache_file_span(f, offs, size, page_size);
        buf.resize(sp.size + page_size);
        uint8_t * dst = (uint8_t *) GGML_PAD((uintptr_t) buf.data(), page_size);
        moe_cache_file_read(f, dst, sp.size, sp.begin, sp.need);
        return dst + (offs - sp.begin);
    }

    // the index must describe every cached expert tensor as it is in its model file, and a sample of experts must match the model file
    void init_sidecar(const llama_model & model, const char * path) {
        sidecar_index = path;
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error(format("MoE cache direct I/O: cannot open the sidecar index %s", path));
        }
        std::stringstream text;
        text << in.rdbuf();
        llama_moe_cache_sidecar_index idx;
        const std::string err = llama_moe_cache_sidecar_parse(text.str(), idx);
        if (!err.empty()) {
            throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: %s", path, err.c_str()));
        }
        if (idx.alignment % page_size != 0) {
            throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: alignment %zu is not a multiple of the page size %zu", path, idx.alignment, page_size));
        }

        const std::string dir = sidecar_index.find('/') == std::string::npos ? std::string(".") : sidecar_index.substr(0, sidecar_index.rfind('/'));
        const std::string bin = dir + "/" + idx.bin_name;
        if (idx.bin_name.find('/') != std::string::npos) {
            throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: the sidecar must be in the directory of the index", path));
        }
        auto base_name = [](const std::string & p) { return p.find('/') == std::string::npos ? p : p.substr(p.rfind('/') + 1); };

        sidecar_layouts.assign(layers.size(), llama_moe_cache_sidecar_layout());
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            llama_moe_cache_sidecar_layout & lay = sidecar_layouts[il];
            std::vector<std::pair<size_t, size_t>> ranges;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const auto it = idx.tensors.find(w->name);
                if (it == idx.tensors.end()) {
                    throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s has no %s", path, w->name));
                }
                const llama_moe_cache_sidecar_tensor & t = it->second;
                const source & src = sources.at(w);
                if (t.n_expert != w->ne[2] || t.nb2 != w->nb[2] || t.source_offs != src.offs) {
                    throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: %s has %lld experts of %zu bytes at offset %zu, the model has %lld of %zu at %zu",
                        path, w->name, (long long) t.n_expert, t.nb2, t.source_offs, (long long) w->ne[2], w->nb[2], src.offs));
                }
                if (base_name(model.source_paths[src.file]) != idx.source_name || files[src.file].size != idx.source_size) {
                    throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s is of %s (%zu bytes), %s is in %s (%zu bytes)",
                        path, idx.source_name.c_str(), idx.source_size, w->name, model.source_paths[src.file].c_str(), files[src.file].size));
                }
                if (ip == 0) {
                    lay.base  = t.base;
                    lay.block = t.block;
                } else if (t.base != lay.base || t.block != lay.block) {
                    throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: the expert tensors of layer %d have different blocks", path, il));
                }
                lay.part_offs.push_back(t.part_offs);
                lay.part_size.push_back(t.nb2);
                ranges.emplace_back(t.part_offs, t.part_offs + t.nb2);
            }
            std::sort(ranges.begin(), ranges.end());
            for (size_t k = 1; k < ranges.size(); ++k) {
                if (ranges[k].first < ranges[k - 1].second) {
                    throw std::runtime_error(format("MoE cache direct I/O: sidecar index %s: the parts of layer %d overlap", path, il));
                }
            }
        }

        sidecar_file = moe_cache_file_open(bin);
        if (sidecar_file.size != idx.bin_size) {
            throw std::runtime_error(format("MoE cache direct I/O: sidecar %s has %zu bytes, its index %s says %zu", bin.c_str(), sidecar_file.size, path, idx.bin_size));
        }

        // a fixed sample: the last and one random expert of every cached expert tensor, byte for byte against the model file
        std::mt19937 rng(0x5eed);
        std::vector<uint8_t> buf_model;
        std::vector<uint8_t> buf_sidecar;
        size_t n_checked = 0;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const llama_moe_cache_sidecar_layout & lay = sidecar_layouts[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const source & src = sources.at(w);
                const int64_t n_expert = w->ne[2];
                for (int64_t e : { (int64_t) (rng() % (uint32_t) n_expert), n_expert - 1 }) {
                    const uint8_t * a = read_check(files[src.file], src.offs + e*w->nb[2], w->nb[2], buf_model);
                    const uint8_t * b = read_check(sidecar_file, lay.base + e*lay.block + lay.part_offs[ip], w->nb[2], buf_sidecar);
                    if (memcmp(a, b, w->nb[2]) != 0) {
                        throw std::runtime_error(format("MoE cache direct I/O: sidecar %s: expert %lld of %s differs from the model file", bin.c_str(), (long long) e, w->name));
                    }
                    n_checked++;
                }
            }
        }
        sidecar = true;
        LLAMA_LOG_INFO("%s: MoE cache direct I/O: sidecar %s matches the model file in %zu sampled experts\n", __func__, bin.c_str(), n_checked);
    }

    // the fills of layer il from the sidecar: the runs of consecutive experts are read in parallel, whole blocks,
    // then each part is uploaded to the same slot of its bank as from the model files
    // returns the bytes uploaded
    size_t read_upload_sidecar(int32_t il, const group & g) {
        const llama_moe_cache_sidecar_layout & lay = sidecar_layouts[il];
        GGML_ASSERT(lay.part_size.size() == g.banks.size());
        llama_moe_cache_sidecar_plan(lay, fills.data(), fills.size(), max_staging, sc_jobs, sc_uploads, sc_waves);

        size_t bytes = 0;
        for (const llama_moe_cache_sidecar_wave & wave : sc_waves) {
            if (staging.size() < wave.arena + page_size) {
                staging.resize(wave.arena + page_size);
            }
            uint8_t * base = (uint8_t *) GGML_PAD((uintptr_t) staging.data(), page_size);
            io_reads.clear();
            for (size_t j = wave.job_begin; j < wave.job_end; ++j) {
                const llama_moe_cache_sidecar_job & job = sc_jobs[j];
                // whole blocks of the sidecar, page-aligned and inside of it (checked by the index parser)
                io_reads.push_back({ sidecar_file.fd, base + job.staging_offs, job.size, job.offs, job.size });
                dio_reads++;
                dio_bytes += job.size;
            }
            io_res.assign(io_reads.size(), 0);
            io_pool->run(io_reads.data(), io_reads.size(), io_res.data());
            // every read has ended: nothing is uploaded unless all of them succeeded
            for (size_t k = 0; k < io_reads.size(); ++k) {
                if (io_res[k] != 0) {
                    GGML_ABORT("MoE cache direct I/O: read of %zu bytes at offset %zu failed: %s", io_reads[k].size, io_reads[k].offs, io_res[k] < 0 ? "end of file" : strerror(io_res[k]));
                }
            }
            // synchronous like the serial path; the backend is idle since the synchronize in copy
            for (size_t k = wave.upload_begin; k < wave.upload_end; ++k) {
                const llama_moe_cache_sidecar_upload & u = sc_uploads[k];
                ggml_backend_tensor_set(g.banks[u.part], base + u.staging_offs, u.bank_offs, u.size);
                bytes += u.size;
            }
        }
        io_reads.clear();
        return bytes;
    }

    // read n experts of w from expert e on, returns their data in the staging buffer
    const uint8_t * read_experts(const ggml_tensor * w, int64_t e, int64_t n) {
        const source & src = sources.at(w);
        const moe_cache_file & f = files[src.file];
        const size_t first = src.offs + e*w->nb[2];
        const moe_cache_span sp = moe_cache_file_span(f, first, n*w->nb[2], page_size);
        if (staging.size() < sp.size + page_size) {
            staging.resize(sp.size + page_size);
        }
        uint8_t * buf = (uint8_t *) GGML_PAD((uintptr_t) staging.data(), page_size);
        moe_cache_file_read(f, buf, sp.size, sp.begin, sp.need);
        dio_reads++;
        dio_bytes += sp.need;
        return buf + (first - sp.begin);
    }

    // the fills of all expert tensors of layer l, read in parallel and then uploaded in the order of the serial loop in copy
    // returns the bytes uploaded
    size_t read_upload_parallel(const layer & l, const group & g) {
        io_reads.clear();
        io_uploads.clear();
        size_t arena = 0;
        size_t bytes = 0;

        auto flush = [&]() {
            if (io_reads.empty()) {
                return;
            }
            if (staging.size() < arena + page_size) {
                staging.resize(arena + page_size);
            }
            uint8_t * base = (uint8_t *) GGML_PAD((uintptr_t) staging.data(), page_size);
            for (size_t k = 0; k < io_reads.size(); ++k) {
                io_reads[k].dst = base + io_uploads[k].staging_offs;
            }
            io_res.assign(io_reads.size(), 0);
            io_pool->run(io_reads.data(), io_reads.size(), io_res.data());
            // every read has ended: nothing is uploaded unless all of them succeeded
            for (size_t k = 0; k < io_reads.size(); ++k) {
                if (io_res[k] != 0) {
                    GGML_ABORT("MoE cache direct I/O: read of %zu bytes at offset %zu failed: %s", io_reads[k].size, io_reads[k].offs, io_res[k] < 0 ? "end of file" : strerror(io_res[k]));
                }
            }
            // synchronous like the serial path; the backend is idle since the synchronize in copy
            for (const upload & u : io_uploads) {
                ggml_backend_tensor_set(u.bank, base + u.staging_offs + u.skip, u.bank_offs, u.size);
            }
            io_reads.clear();
            io_uploads.clear();
            arena = 0;
        };

        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const source & src = sources.at(w);
            const moe_cache_file & f = files[src.file];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                // the same page-aligned range as read_experts
                const size_t first = src.offs + fills[i].expert*w->nb[2];
                const moe_cache_span sp = moe_cache_file_span(f, first, n*w->nb[2], page_size);
                const size_t size  = sp.size;
                if (!io_reads.empty() && arena + size > max_staging) {
                    flush();
                }
                io_reads.push_back({ f.fd, nullptr, size, sp.begin, sp.need });
                io_uploads.push_back({ bank, fills[i].slot*expert_size, n*expert_size, arena, first - sp.begin });
                arena += GGML_PAD(size, page_size);
                dio_reads++;
                dio_bytes += sp.need;
                bytes += n*expert_size;
                i += n;
            }
        }
        flush();
        return bytes;
    }

    // direct I/O: every ubatch of up to n_tokens tokens must read the cache of every cached layer
    void require_batch(int64_t n_tokens) const {
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const int64_t need = std::min(n_tokens*(int64_t) n_expert_used, l.slot_map->ne[1]);
            if (need > groups[l.ig].n_slots) {
                throw std::runtime_error(format("MoE cache direct I/O: a ubatch of %lld tokens can select %lld experts of layer %d, more than the %d slots of its cache, "
                    "it would read the mapped experts (raise --moe-cache-mib or lower -ub)", (long long) n_tokens, (long long) need, il, (int) groups[l.ig].n_slots));
            }
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        if (direct_io && sidecar) {
            bytes = read_upload_sidecar(il, g);
        } else if (direct_io && io_pool) {
            bytes = read_upload_parallel(l, g);
        } else
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                if (direct_io) {
                    // synchronous, the staging buffer is reused; the backend is idle since the synchronize above
                    ggml_backend_tensor_set(bank, read_experts(w, fills[i].expert, n), fills[i].slot*expert_size, n*expert_size);
                } else {
                    ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                }
                bytes += n*expert_size;
                i += n;
            }
        }

        if (il == il_trace) {
            LLAMA_LOG_INFO("llama_moe_cache: direct I/O trace: t_us = %lld, read = %zu\n", (long long) moe_cache_time_us(), dio_bytes);
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (direct_io) {
            LLAMA_LOG_INFO("llama_moe_cache: direct I/O: reads = %zu, read = %.2f MiB (%zu bytes)\n", dio_reads, dio_bytes/1024.0/1024.0, dio_bytes);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

void llama_moe_cache::require_batch(int64_t n_tokens) const {
    pimpl->require_batch(n_tokens);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
