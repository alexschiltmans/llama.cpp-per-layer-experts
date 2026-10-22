#pragma once

// positional reads of the MoE cache's direct I/O, run in parallel by a small persistent pool,
// and the reads and uploads of the missing experts of a layer from an expert-major sidecar

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// an expert tensor in the index of an expert-major sidecar
struct llama_moe_cache_sidecar_tensor {
    int64_t n_expert;
    size_t  nb2;
    size_t  base;        // of the layer in the sidecar
    size_t  block;       // bytes per expert of the layer, all of its expert tensors
    size_t  part_offs;   // of this tensor in a block
    size_t  source_offs; // of the tensor data in the model file
};

struct llama_moe_cache_sidecar_index {
    size_t      alignment   = 0;
    std::string source_name;     // file name of the model file
    size_t      source_size = 0;
    std::string bin_name;        // file name of the sidecar, in the directory of the index
    size_t      bin_size    = 0;
    std::map<std::string, llama_moe_cache_sidecar_tensor> tensors;
};

// parses the text index written by the sidecar builder, returns an empty string or the error
//   moe-experts-major 1
//   alignment <bytes>
//   source <file name> <size>
//   bin <file name> <size>
//   tensor <name> <n_expert> <nb2> <layer base> <block size> <part offset> <source offset> <sha256>
//   end
// lines starting with # are comments
static inline std::string llama_moe_cache_sidecar_parse(const std::string & text, llama_moe_cache_sidecar_index & idx) {
    idx = llama_moe_cache_sidecar_index();
    std::istringstream in(text);
    std::string line;
    size_t n_line = 0;
    bool has_version = false;
    bool has_end     = false;

    auto num = [](const std::string & tok, size_t & res) {
        if (tok.empty() || tok.size() > 19 || tok.find_first_not_of("0123456789") != std::string::npos) {
            return false;
        }
        res = (size_t) strtoull(tok.c_str(), nullptr, 10);
        return true;
    };

    while (std::getline(in, line)) {
        n_line++;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (has_end) {
            return "line " + std::to_string(n_line) + ": text after end";
        }
        std::istringstream ls(line);
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) {
            tok.push_back(t);
        }
        const std::string at = "line " + std::to_string(n_line) + ": ";
        if (!has_version) {
            if (tok.size() != 2 || tok[0] != "moe-experts-major" || tok[1] != "1") {
                return at + "not an expert-major sidecar index of format 1";
            }
            has_version = true;
        } else if (tok[0] == "alignment" && tok.size() == 2) {
            if (!num(tok[1], idx.alignment) || idx.alignment == 0 || (idx.alignment & (idx.alignment - 1)) != 0) {
                return at + "bad alignment";
            }
        } else if (tok[0] == "source" && tok.size() == 3) {
            idx.source_name = tok[1];
            if (!num(tok[2], idx.source_size)) {
                return at + "bad source size";
            }
        } else if (tok[0] == "bin" && tok.size() == 3) {
            idx.bin_name = tok[1];
            if (!num(tok[2], idx.bin_size)) {
                return at + "bad bin size";
            }
        } else if (tok[0] == "tensor" && tok.size() == 9) {
            llama_moe_cache_sidecar_tensor t;
            size_t n_expert = 0;
            if (!num(tok[2], n_expert) || !num(tok[3], t.nb2) || !num(tok[4], t.base) || !num(tok[5], t.block) ||
                !num(tok[6], t.part_offs) || !num(tok[7], t.source_offs) || n_expert == 0 || n_expert > (1 << 20) || t.nb2 == 0) {
                return at + "bad tensor line";
            }
            t.n_expert = (int64_t) n_expert;
            if (idx.alignment == 0 || idx.bin_size == 0) {
                return at + "tensor before the alignment and bin lines";
            }
            if (t.base % idx.alignment != 0 || t.block % idx.alignment != 0 || t.part_offs % idx.alignment != 0 ||
                t.part_offs >= t.block || t.nb2 > t.block - t.part_offs ||
                t.block > (idx.bin_size - std::min(t.base, idx.bin_size))/n_expert || t.base >= idx.bin_size) {
                return at + "tensor " + tok[1] + " is not aligned or outside of the sidecar";
            }
            if (!idx.tensors.emplace(tok[1], t).second) {
                return at + "duplicate tensor " + tok[1];
            }
        } else if (tok[0] == "end" && tok.size() == 1) {
            has_end = true;
        } else {
            return at + "unknown line";
        }
    }
    if (!has_version || !has_end || idx.alignment == 0 || idx.source_name.empty() || idx.bin_name.empty()) {
        return "incomplete index";
    }
    return "";
}

// expert-major sidecar of a layer: expert e of expert tensor ip is at base + e*block + part_offs[ip], part_size[ip] bytes
struct llama_moe_cache_sidecar_layout {
    size_t base  = 0;
    size_t block = 0;
    std::vector<size_t> part_offs; // in the order of the banks
    std::vector<size_t> part_size; // nb[2] of each expert tensor
};

struct llama_moe_cache_fill {
    int32_t expert;
    int32_t slot;
};

// the whole blocks of a run of consecutive experts, read into the staging buffer of its wave at staging_offs
struct llama_moe_cache_sidecar_job {
    size_t offs;
    size_t size;
    size_t staging_offs;
};

// part of one expert, from the staging buffer of its wave to the bank of the expert tensor
struct llama_moe_cache_sidecar_upload {
    uint32_t part;
    size_t   bank_offs;
    size_t   size;
    size_t   staging_offs;
};

// jobs [job_begin, job_end) are read in parallel into a staging buffer of arena bytes, then uploads [upload_begin, upload_end) are done
struct llama_moe_cache_sidecar_wave {
    size_t job_begin;
    size_t job_end;
    size_t upload_begin;
    size_t upload_end;
    size_t arena;
};

// plans the reads and uploads of the fills of a layer, sorted by expert as the cache plans them:
// one job per run of consecutive experts, waves of at most max_staging bytes (a larger job is a wave of its own),
// and within a wave the uploads of each expert tensor in turn, expert by expert in the order of the fills,
// each to slot*part_size of its bank as the reads from the model files do
// the base and block of the layout are multiples of the page size, so are all job offsets and sizes
static inline void llama_moe_cache_sidecar_plan(
        const llama_moe_cache_sidecar_layout & lay, const llama_moe_cache_fill * fills, size_t n_fills, size_t max_staging,
        std::vector<llama_moe_cache_sidecar_job> & jobs, std::vector<llama_moe_cache_sidecar_upload> & uploads,
        std::vector<llama_moe_cache_sidecar_wave> & waves) {
    jobs.clear();
    uploads.clear();
    waves.clear();

    std::vector<size_t> staging_of(n_fills); // offset of the block of each fill in the staging buffer of its wave
    size_t arena      = 0;
    size_t job_begin  = 0;
    size_t fill_begin = 0; // first fill of the wave

    auto close_wave = [&](size_t fill_end) {
        if (jobs.size() == job_begin) {
            return;
        }
        const size_t upload_begin = uploads.size();
        for (size_t ip = 0; ip < lay.part_offs.size(); ++ip) {
            for (size_t i = fill_begin; i < fill_end; ++i) {
                uploads.push_back({ (uint32_t) ip, (size_t) fills[i].slot*lay.part_size[ip], lay.part_size[ip], staging_of[i] + lay.part_offs[ip] });
            }
        }
        waves.push_back({ job_begin, jobs.size(), upload_begin, uploads.size(), arena });
        job_begin  = jobs.size();
        fill_begin = fill_end;
        arena      = 0;
    };

    // any order of the fills is planned correctly, sorted fills give the longest runs
    for (size_t i = 0; i < n_fills;) {
        size_t n = 1;
        while (i + n < n_fills && fills[i + n].expert == fills[i].expert + (int32_t) n) {
            n++;
        }
        const size_t size = n*lay.block;
        if (jobs.size() > job_begin && arena + size > max_staging) {
            close_wave(i);
        }
        jobs.push_back({ lay.base + (size_t) fills[i].expert*lay.block, size, arena });
        for (size_t k = 0; k < n; ++k) {
            staging_of[i + k] = arena + k*lay.block;
        }
        arena += size;
        i += n;
    }
    close_wave(n_fills);
}

#if !defined(_WIN32)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <unistd.h>

struct llama_moe_cache_read {
    int       fd;
    uint8_t * dst;
    size_t    size;
    size_t    offs;
    size_t    need = 0; // the read may stop short after need bytes (O_DIRECT at the end of the file), 0: all size bytes
};

// read size bytes at offs, retrying on EINTR; a read that stops short is accepted once need bytes are read
// (need == 0 or need >= size: all size bytes)
// returns 0, the errno of a failed read, or -1 at an early end of file
static inline int llama_moe_cache_pread(int fd, uint8_t * dst, size_t size, size_t offs, size_t need = 0) {
    if (need == 0 || need > size) {
        need = size;
    }
    while (size > 0) {
        const ssize_t n = pread(fd, dst, std::min(size, (size_t) 1 << 30), (off_t) offs);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            return errno != 0 ? errno : EIO;
        }
        if (n == 0) {
            return -1;
        }
        dst  += n;
        size -= n;
        offs += n;
        // no further read at the end of the file: with O_DIRECT its offset is not aligned
        if ((size_t) n >= need) {
            return 0;
        }
        need -= n;
    }
    return 0;
}

// n_threads reads at a time: the calling thread and n_threads - 1 workers that wait between calls
class llama_moe_cache_read_pool {
public:
    explicit llama_moe_cache_read_pool(int n_threads) {
        for (int i = 1; i < n_threads; ++i) {
            workers.emplace_back([this] { work(); });
        }
    }

    ~llama_moe_cache_read_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_start.notify_all();
        for (std::thread & t : workers) {
            t.join();
        }
    }

    llama_moe_cache_read_pool(const llama_moe_cache_read_pool &) = delete;
    llama_moe_cache_read_pool & operator=(const llama_moe_cache_read_pool &) = delete;

    int n_threads() const {
        return (int) workers.size() + 1;
    }

    // runs all n reads and returns when every one has ended; res[i] is the result of llama_moe_cache_pread for read i
    void run(const llama_moe_cache_read * reads, size_t n, int * res) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs   = reads;
            n_jobs = n;
            out    = res;
            next.store(0);
            busy   = workers.size();
            gen++;
        }
        cv_start.notify_all();
        drain(reads, n, res);
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [this] { return busy == 0; });
        jobs   = nullptr;
        n_jobs = 0;
        out    = nullptr;
    }

private:
    void drain(const llama_moe_cache_read * reads, size_t n, int * res) {
        for (size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
            const llama_moe_cache_read & r = reads[i];
            res[i] = llama_moe_cache_pread(r.fd, r.dst, r.size, r.offs, r.need);
        }
    }

    void work() {
        uint64_t seen = 0;
        while (true) {
            const llama_moe_cache_read * reads;
            size_t n;
            int * res;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_start.wait(lock, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen  = gen;
                reads = jobs;
                n     = n_jobs;
                res   = out;
            }
            drain(reads, n, res);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--busy == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    bool     stop = false;
    uint64_t gen  = 0;
    size_t   busy = 0;
    const llama_moe_cache_read * jobs = nullptr;
    size_t n_jobs = 0;
    int *  out    = nullptr;
    std::atomic<size_t> next{0};
};

#endif
