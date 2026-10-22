// the parallel reads of the MoE cache's direct I/O return the same bytes as serial reads,
// and the reads and uploads planned from an expert-major sidecar put the same bytes in the same slots as the reads from the model file

#include "../src/llama-moe-cache-io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static int fail(const char * msg) {
    fprintf(stderr, "test-moe-cache-io: FAIL: %s\n", msg);
    return 1;
}

static int test_sidecar_parse() {
    const std::string head = "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\n";
    const std::string good = head + "# comment\ntensor a 2 100 0 32768 0 64 x\ntensor b 2 16384 0 32768 16384 9000 x\nend\n";
    llama_moe_cache_sidecar_index idx;
    if (!llama_moe_cache_sidecar_parse(good, idx).empty() || idx.tensors.size() != 2 || idx.alignment != 16384 ||
        idx.source_name != "m.gguf" || idx.source_size != 1000 || idx.bin_name != "m.bin" || idx.bin_size != 98304 ||
        idx.tensors["b"].part_offs != 16384 || idx.tensors["b"].source_offs != 9000 || idx.tensors["a"].n_expert != 2) {
        return fail("sidecar index not parsed");
    }
    const char * bad[] = {
        "moe-experts-major 2\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\nend\n", // format
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 100 0 32768 0 64 x\n", // no end
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 100 4096 32768 0 64 x\nend\n", // base not aligned
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 100 0 30000 0 64 x\nend\n", // block not aligned
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 20000 0 32768 16384 64 x\nend\n", // part past the block
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 3 100 32768 32768 0 64 x\nend\n", // past the sidecar
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 100 0 32768 0 64 x\ntensor a 2 100 0 32768 0 64 x\nend\n", // duplicate
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\ntensor a 2 -100 0 32768 0 64 x\nend\n", // number
        "moe-experts-major 1\nalignment 12288\nsource m.gguf 1000\nbin m.bin 98304\nend\n", // alignment
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nend\n", // no bin
        "moe-experts-major 1\nalignment 16384\nsource m.gguf 1000\nbin m.bin 98304\nend\ntensor a 2 100 0 32768 0 64 x\n", // after end
    };
    for (const char * text : bad) {
        if (llama_moe_cache_sidecar_parse(text, idx).empty()) {
            fprintf(stderr, "accepted:\n%s", text);
            return fail("bad sidecar index accepted");
        }
    }
    return 0;
}

// a model file with the expert tensors of one layer and its expert-major sidecar, written like the builder writes it;
// random fills are planned, read with the pool and uploaded to host banks, which must equal the banks filled from the model file
static int test_sidecar_plan(std::mt19937_64 & rng) {
    const size_t align    = 16384;
    const int    n_expert = 37;
    const size_t nb2[3]   = { 5000, 7000, 12345 }; // gate, up, down; not page multiples
    const size_t n_parts  = 3;
    const size_t page     = (size_t) sysconf(_SC_PAGESIZE);
    if (align % page != 0) {
        return fail("page size");
    }

    // model file: some header bytes, then the tensors at unaligned offsets
    std::vector<size_t> src_offs(n_parts);
    size_t model_size = 777;
    for (size_t ip = 0; ip < n_parts; ++ip) {
        src_offs[ip] = model_size;
        model_size  += nb2[ip]*n_expert + 32;
    }
    std::vector<uint8_t> model(model_size);
    for (auto & b : model) {
        b = (uint8_t) rng();
    }

    // sidecar: one layer at base 2*align (as if after another layer), each part padded to align
    llama_moe_cache_sidecar_layout lay;
    lay.base = 2*align;
    for (size_t ip = 0; ip < n_parts; ++ip) {
        lay.part_offs.push_back(lay.block);
        lay.part_size.push_back(nb2[ip]);
        lay.block += (nb2[ip] + align - 1)/align*align;
    }
    std::vector<uint8_t> side(lay.base + lay.block*n_expert, 0);
    for (int e = 0; e < n_expert; ++e) {
        for (size_t ip = 0; ip < n_parts; ++ip) {
            memcpy(side.data() + lay.base + e*lay.block + lay.part_offs[ip], model.data() + src_offs[ip] + e*nb2[ip], nb2[ip]);
        }
    }

    // the index of the sidecar goes through the parser
    std::string text = "moe-experts-major 1\nalignment 16384\nsource m.gguf " + std::to_string(model_size) + "\nbin m.bin " + std::to_string(side.size()) + "\n";
    const char * names[3] = { "gate", "up", "down" };
    for (size_t ip = 0; ip < n_parts; ++ip) {
        text += std::string("tensor ") + names[ip] + " " + std::to_string(n_expert) + " " + std::to_string(nb2[ip]) + " " + std::to_string(lay.base) + " " +
            std::to_string(lay.block) + " " + std::to_string(lay.part_offs[ip]) + " " + std::to_string(src_offs[ip]) + " x\n";
    }
    text += "end\n";
    llama_moe_cache_sidecar_index idx;
    if (!llama_moe_cache_sidecar_parse(text, idx).empty()) {
        return fail("sidecar index of the test not parsed");
    }
    llama_moe_cache_sidecar_layout lay_idx;
    for (size_t ip = 0; ip < n_parts; ++ip) {
        const auto & t = idx.tensors.at(names[ip]);
        lay_idx.base  = t.base;
        lay_idx.block = t.block;
        lay_idx.part_offs.push_back(t.part_offs);
        lay_idx.part_size.push_back(t.nb2);
    }
    if (lay_idx.base != lay.base || lay_idx.block != lay.block || lay_idx.part_offs != lay.part_offs || lay_idx.part_size != lay.part_size) {
        return fail("sidecar layout differs from its index");
    }

    char path[] = "/tmp/test-moe-cache-sidecar-XXXXXX";
    const int fd_w = mkstemp(path);
    if (fd_w < 0) {
        return fail("mkstemp");
    }
    if (write(fd_w, side.data(), side.size()) != (ssize_t) side.size()) {
        unlink(path);
        return fail("write");
    }
    close(fd_w);
    const int fd = open(path, O_RDONLY);
    unlink(path);
    if (fd < 0) {
        return fail("open");
    }
#if defined(__APPLE__)
    fcntl(fd, F_NOCACHE, 1);
#endif

    const int n_slots = 24;
    int rc = 0;
    std::vector<llama_moe_cache_sidecar_job> jobs;
    std::vector<llama_moe_cache_sidecar_upload> uploads;
    std::vector<llama_moe_cache_sidecar_wave> waves;
    for (int n_threads : { 1, 3, 8 }) {
        llama_moe_cache_read_pool pool(n_threads);
        for (int round = 0; round < 300 && rc == 0; ++round) {
            // distinct experts sorted as the cache plans them, distinct slots in any order
            std::vector<int> experts(n_expert);
            std::iota(experts.begin(), experts.end(), 0);
            std::shuffle(experts.begin(), experts.end(), rng);
            size_t n_fills = 1 + rng() % n_slots;
            if (round == 0) {
                n_fills = 1;
                experts[0] = n_expert - 1; // the last expert alone
            }
            experts.resize(n_fills);
            if (round == 1) {
                for (size_t i = 0; i < n_fills; ++i) {
                    experts[i] = n_expert - (int) n_fills + (int) i; // one run to the end
                }
            }
            std::sort(experts.begin(), experts.end());
            std::vector<int> slots(n_slots);
            std::iota(slots.begin(), slots.end(), 0);
            std::shuffle(slots.begin(), slots.end(), rng);
            std::vector<llama_moe_cache_fill> fills(n_fills);
            for (size_t i = 0; i < n_fills; ++i) {
                fills[i] = { experts[i], round % 3 == 0 ? (int32_t) i : slots[i] };
            }
            // small budgets split the reads into waves
            const size_t max_staging = round % 2 == 0 ? (size_t) 256 << 20 : lay.block*(1 + rng() % 4);
            llama_moe_cache_sidecar_plan(lay, fills.data(), fills.size(), max_staging, jobs, uploads, waves);

            // one job per run of consecutive experts, page aligned, whole blocks
            size_t n_runs = 1;
            for (size_t i = 1; i < n_fills; ++i) {
                n_runs += fills[i].expert != fills[i - 1].expert + 1;
            }
            if (jobs.size() != n_runs || uploads.size() != n_fills*n_parts) {
                rc = fail("sidecar plan: number of jobs or uploads");
                break;
            }
            for (const auto & j : jobs) {
                if (j.offs % page != 0 || j.size % lay.block != 0 || j.offs < lay.base || (j.offs - lay.base) % lay.block != 0 || j.offs + j.size > side.size()) {
                    rc = fail("sidecar plan: job not aligned or outside of the sidecar");
                }
            }

            std::vector<std::vector<uint8_t>> banks(n_parts), ref(n_parts);
            for (size_t ip = 0; ip < n_parts; ++ip) {
                banks[ip].assign(nb2[ip]*n_slots, 0xab);
                ref[ip].assign(nb2[ip]*n_slots, 0xab);
                // the reads from the model file: expert e of tensor ip to slot s
                for (const auto & f : fills) {
                    memcpy(ref[ip].data() + f.slot*nb2[ip], model.data() + src_offs[ip] + f.expert*nb2[ip], nb2[ip]);
                }
            }
            size_t prev_part = 0;
            for (const auto & w : waves) {
                if (w.job_end - w.job_begin > 1 && w.arena > max_staging) {
                    rc = fail("sidecar plan: wave over budget");
                }
                std::vector<uint8_t> staging(w.arena + page);
                uint8_t * base = (uint8_t *) (((uintptr_t) staging.data() + page - 1) & ~(uintptr_t) (page - 1));
                std::vector<llama_moe_cache_read> reads;
                for (size_t j = w.job_begin; j < w.job_end; ++j) {
                    reads.push_back({ fd, base + jobs[j].staging_offs, jobs[j].size, jobs[j].offs });
                }
                std::vector<int> res(reads.size(), 123);
                pool.run(reads.data(), reads.size(), res.data());
                for (int r : res) {
                    if (r != 0) {
                        rc = fail("sidecar read failed");
                    }
                }
                prev_part = 0;
                for (size_t k = w.upload_begin; k < w.upload_end; ++k) {
                    const auto & u = uploads[k];
                    if (u.part < prev_part) {
                        rc = fail("sidecar plan: uploads not in tensor order");
                    }
                    prev_part = u.part;
                    if (u.staging_offs + u.size > w.arena || u.bank_offs + u.size > banks[u.part].size()) {
                        rc = fail("sidecar plan: upload out of range");
                        break;
                    }
                    memcpy(banks[u.part].data() + u.bank_offs, base + u.staging_offs, u.size);
                }
            }
            if (rc == 0 && banks != ref) {
                rc = fail("sidecar uploads differ from the reads of the model file");
            }
        }
    }
    close(fd);
    return rc;
}

#if defined(__linux__)
// O_DIRECT reads of whole pages from a file in the working directory whose size is not a multiple of the page size,
// as the cache reads the model files on Linux; skipped where the filesystem has no O_DIRECT (e.g. tmpfs on old kernels)
static int test_o_direct(std::mt19937_64 & rng) {
    const size_t page = std::max((size_t) sysconf(_SC_PAGESIZE), (size_t) 4096);
    const size_t file_size = ((size_t) 4 << 20) - 100;
    std::vector<uint8_t> data(file_size);
    for (auto & b : data) {
        b = (uint8_t) rng();
    }
    char path[] = "test-moe-cache-io-direct-XXXXXX";
    const int fd_w = mkstemp(path);
    if (fd_w < 0) {
        return fail("mkstemp in the working directory");
    }
    if (write(fd_w, data.data(), data.size()) != (ssize_t) data.size() || fsync(fd_w) != 0) {
        close(fd_w);
        unlink(path);
        return fail("write");
    }
    close(fd_w);
    const int fd = open(path, O_RDONLY | O_DIRECT);
    unlink(path);
    if (fd < 0) {
        printf("test-moe-cache-io: O_DIRECT not supported here (%s), skipped\n", strerror(errno));
        return 0;
    }
    void * mem = nullptr;
    if (posix_memalign(&mem, page, file_size + 2*page) != 0) {
        close(fd);
        return fail("posix_memalign");
    }
    uint8_t * buf = (uint8_t *) mem;
    int rc = 0;
    bool skipped = false;
    llama_moe_cache_read_pool pool(4);
    for (int round = 0; round < 200 && rc == 0; ++round) {
        // bytes [first, first + n) as page-aligned spans, the last pages past the end of the file
        const size_t first = rng() % file_size;
        const size_t n     = 1 + rng() % (file_size - first);
        const size_t begin = first & ~(page - 1);
        const size_t end   = (first + n + page - 1) & ~(page - 1);
        const size_t need  = std::min(end, file_size) - begin;
        llama_moe_cache_read r = { fd, buf, end - begin, begin, need };
        int res = 123;
        pool.run(&r, 1, &res);
        if (res == EINVAL && round == 0) {
            printf("test-moe-cache-io: O_DIRECT reads fail here (EINVAL), skipped\n");
            skipped = true;
            break;
        }
        if (res != 0 || memcmp(buf + (first - begin), data.data() + first, n) != 0) {
            rc = fail("O_DIRECT read");
        }
        if (end > file_size && rc == 0) {
            // one byte more than the file has is an early end of file
            if (llama_moe_cache_pread(fd, buf, end - begin, begin, need + 1) != -1) {
                rc = fail("O_DIRECT early end of file not reported");
            }
        }
    }
    if (rc == 0 && !skipped) {
        printf("test-moe-cache-io: O_DIRECT reads OK (page %zu)\n", page);
    }
    free(mem);
    close(fd);
    return rc;
}
#endif

int main() {
    std::mt19937_64 rng_sc(7);
    if (test_sidecar_parse() != 0 || test_sidecar_plan(rng_sc) != 0) {
        return 1;
    }
#if defined(__linux__)
    if (test_o_direct(rng_sc) != 0) {
        return 1;
    }
#endif

    char path[] = "/tmp/test-moe-cache-io-XXXXXX";
    const int fd_w = mkstemp(path);
    if (fd_w < 0) {
        return fail("mkstemp");
    }
    const size_t file_size = (size_t) 24 << 20;
    std::vector<uint8_t> data(file_size);
    std::mt19937_64 rng(42);
    for (auto & b : data) {
        b = (uint8_t) rng();
    }
    if (write(fd_w, data.data(), data.size()) != (ssize_t) data.size()) {
        unlink(path);
        return fail("write");
    }
    close(fd_w);
    const int fd = open(path, O_RDONLY);
    unlink(path);
    if (fd < 0) {
        return fail("open");
    }
#if defined(__APPLE__)
    fcntl(fd, F_NOCACHE, 1);
#endif

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    int rc = 0;
    for (int n_threads : { 1, 2, 8, 32 }) {
        llama_moe_cache_read_pool pool(n_threads);
        for (int round = 0; round < 50 && rc == 0; ++round) {
            // page-aligned ranges of up to 2 MiB, some ending at the end of the file, as the cache reads them
            const size_t n = 1 + rng() % 64;
            std::vector<llama_moe_cache_read> reads(n);
            std::vector<std::vector<uint8_t>> par(n), ser(n);
            for (size_t i = 0; i < n; ++i) {
                const size_t begin = (rng() % (file_size/page))*page;
                size_t size = 1 + rng() % ((size_t) 2 << 20);
                if (rng() % 8 == 0 || begin + size > file_size) {
                    size = file_size - begin;
                }
                par[i].assign(size, 0);
                ser[i].assign(size, 0);
                reads[i] = { fd, par[i].data(), size, begin };
            }
            std::vector<int> res(n, 123);
            pool.run(reads.data(), n, res.data());
            for (size_t i = 0; i < n; ++i) {
                if (res[i] != 0) {
                    rc = fail("parallel read failed");
                    break;
                }
                if (llama_moe_cache_pread(fd, ser[i].data(), reads[i].size, reads[i].offs) != 0) {
                    rc = fail("serial read failed");
                    break;
                }
                if (memcmp(par[i].data(), ser[i].data(), reads[i].size) != 0 || memcmp(par[i].data(), data.data() + reads[i].offs, reads[i].size) != 0) {
                    rc = fail("bytes differ");
                    break;
                }
            }
        }
        // a read past the end of the file is reported for that read only, the others still complete
        if (rc == 0) {
            std::vector<uint8_t> a(page), b(2*page);
            llama_moe_cache_read reads[2] = { { fd, a.data(), page, 0 }, { fd, b.data(), 2*page, file_size - page } };
            int res[2] = { 123, 123 };
            pool.run(reads, 2, res);
            if (res[0] != 0 || res[1] != -1 || memcmp(a.data(), data.data(), page) != 0) {
                rc = fail("end of file not reported");
            }
        }
        // O_DIRECT reads whole pages: a read past the end of the file is accepted once the bytes up to it are read
        if (rc == 0) {
            std::vector<uint8_t> a(2*page, 0), b(2*page, 0), c(2*page, 0);
            llama_moe_cache_read reads[3] = {
                { fd, a.data(), 2*page, file_size - page, page },     // stops at the end of the file, after the needed page
                { fd, b.data(), 2*page, file_size - page, page + 1 }, // needs a byte past the end of the file
                { fd, c.data(), 2*page, file_size - 2*page, page },   // not at the end: all of it is read
            };
            int res[3] = { 123, 123, 123 };
            pool.run(reads, 3, res);
            if (res[0] != 0 || memcmp(a.data(), data.data() + file_size - page, page) != 0 || res[1] != -1 ||
                res[2] != 0 || memcmp(c.data(), data.data() + file_size - 2*page, 2*page) != 0) {
                rc = fail("short read at the end of the file not handled");
            }
            if (llama_moe_cache_pread(fd, b.data(), 2*page, file_size - page, page) != 0 ||
                llama_moe_cache_pread(fd, b.data(), 2*page, file_size - page, 2*page) != -1 ||
                llama_moe_cache_pread(fd, b.data(), 2*page, file_size - page) != -1) {
                rc = fail("serial short read at the end of the file not handled");
            }
        }
    }
    close(fd);
    if (rc == 0) {
        printf("test-moe-cache-io: OK\n");
    }
    return rc;
}
