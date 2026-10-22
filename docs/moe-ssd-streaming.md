# SSD streaming of MoE experts (macOS)

This mode keeps a MoE model's routed experts on the SSD; only a bounded GPU cache of them sits in memory.
Upstream, `--cpu-moe` maps the experts from the file and `--moe-cache-mib` caches selected ones on the GPU, but misses
read the mapped file and grow the page cache.

With `LLAMA_MOE_CACHE_DIRECT_IO=1`, the cache reads its misses from the model file with the page cache turned off
(`F_NOCACHE`), straight into the cache slots. The reads of a layer run on a pool of threads; the uploads keep the
serial order, so the cache receives the same bytes as without direct I/O.

A GGUF file stores gate, up and down tensors separately (three reads per missed expert); the sidecar stores each
expert's three parts together, so consecutive missed experts take one read. The sidecar is optional, but the path
without it is not measured; all results below use the sidecar.

Tested with one model (Qwen3.8-Flash-Next-Coder cut240 GGUF) on one M4 Pro (48 GB), macOS 26.6, with a Metal build.
Streaming was measured only with `-c 4096 -ub 2 -b 2`.

## Requirements

- macOS on Apple Silicon, built with Metal:

  ```sh
  git clone --branch per-layer-experts https://github.com/alexschiltmans/llama.cpp-per-layer-experts llama.cpp-per-layer-experts
  cd llama.cpp-per-layer-experts
  git checkout e2d5c9f91d574b22f46455037bc29b27665cc60f
  cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j
  ```

- The model loaded with mmap (the default) and `-nr` (`--no-repack`): direct I/O reads each expert tensor at its
  offset in the model file.
- `--cpu-moe` and `-ngl 99`: every layer runs on the GPU, its experts stay file-backed (mapped from the model file),
  and the cache holds the selected ones.
- A `--moe-cache-mib` that caches every layer: each group of layers needs at least as many slots as the model
  routes experts per token.
- A small `-ub` (and `-b`): a ubatch of n tokens can select n times the routed experts of a layer, and they must fit
  in the slots of its cache.
- For the sidecar: all routed expert tensors in one GGUF file (for a split model, one shard), and Python 3 with
  numpy to build it.

## Building the sidecar

```sh
python3 -I scripts/build-moe-sidecar.py Qwen3.8-Flash-Next-Coder-cut240-IQ1_M-00002-of-00003.gguf
```

Pass the file that holds the expert tensors. The script needs Python 3 with numpy. It writes
`<file>.experts-major.bin` and `<file>.experts-major.idx` next to it, and for the model above builds in about 30 s:

- Each part of an expert is padded to 16 KiB, each layer starts at a 16 KiB boundary. For the model above the
  sidecar is 23,898,914,816 bytes, 1.7% more than the shard.
- It builds without the page cache, then checks every part's sha256 and zero padding before writing the index.
- It needs the sidecar's size free on disk (23.9 GB for the model above) plus a headroom: it refuses to start if
  less than 2 GiB would stay free (`--headroom-gib N` changes that). It also refuses to overwrite an existing sidecar.
- `--gguf-py <dir>` selects the gguf-py package; the default is the one in this repository.

The index names the source file and its size. Keep the `.bin` and the `.idx` next to the shard they were built from
(the `00002` shard above) and do not rename any of the three files.

## Environment variables

| Variable | Values | Effect |
| --- | --- | --- |
| `LLAMA_MOE_CACHE_DIRECT_IO` | `1` | Read cache misses from the model file without the page cache. |
| `LLAMA_MOE_CACHE_DIRECT_IO_THREADS` | `1` to `32`, default `8` | Reads in flight per layer (8 if unset); `1` reads one run after the other. |
| `LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR` | path to `<file>.experts-major.idx` | Read misses from the sidecar. Needs `LLAMA_MOE_CACHE_DIRECT_IO=1`. |
| `LLAMA_MOE_CACHE_DIRECT_IO_TRACE` | `1` | Log the bytes read so far after each ubatch, at the last cached layer. |

At the end of a run the log shows `llama_moe_cache: direct I/O: reads = ..., read = ... MiB`, next to the cache's
hit rate.

## What it refuses, and why

Direct I/O only helps if no layer ever computes from the mapped experts: one such read pulls pages of the model
file into the page cache. Setups that could do that stop at init with an error instead:

- A layer with only some of its experts in host memory, or a layer not assigned to a GPU.
- A `--moe-cache-mib` too small for a group of layers (they would not be cached).
- A ubatch size that can select more experts of a layer than its cache has slots ("raise --moe-cache-mib or lower
  -ub").
- Expert data that is not mapped from a model file, as with `--no-mmap` or repacked weights (use `-nr`).
- A model file that cannot be opened without the page cache.
- `LLAMA_MOE_CACHE_DIRECT_IO_THREADS` outside 1 to 32.

With a sidecar, init also refuses an index that does not describe every cached expert tensor exactly as it is in the
model file (count, size, offset, file name and size), a sidecar whose size differs from the index, and a sidecar
that differs from the model file in a sample of experts (two per tensor, compared byte for byte).

## Example

```sh
LLAMA_MOE_CACHE_DIRECT_IO=1 \
LLAMA_MOE_CACHE_DIRECT_IO_THREADS=8 \
LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR=Qwen3.8-Flash-Next-Coder-cut240-IQ1_M-00002-of-00003.gguf.experts-major.idx \
./build/bin/llama-completion -m Qwen3.8-Flash-Next-Coder-cut240-IQ1_M-00001-of-00003.gguf -f prompt.txt -n 512 \
  --temp 0 --top-k 1 -s 1 -no-cnv -nr -fit off -fa on -c 4096 -ub 2 -b 2 -ngl 99 --cpu-moe --moe-cache-mib 12288
```

Without `LLAMA_MOE_CACHE_DIRECT_IO_SIDECAR` the same command reads the misses from the shard itself (not measured;
all results below use the sidecar).

## Measured

Machine: one M4 Pro (48 GB), macOS 26.6, in normal desktop use. Each streaming row: the example above
plus `-t 10 --ignore-eos --no-warmup`, 512 greedy tokens, 3 cold runs; the all-GPU row drops `--cpu-moe`,
`--moe-cache-mib` and the env vars. Streaming was measured only with `-c 4096 -ub 2 -b 2`.

| `--moe-cache-mib` | Decode tokens/s, median (min-max) of 3 cold runs | MB read from SSD per token | Wired, machine peak (GB) | Bitwise identical to all-GPU |
| --- | --- | --- | --- | --- |
| 4096 | 3.97 (3.90-4.00) | 302 | 12.9 | yes |
| 12288 | 10.06 (9.92-10.41) | 60 | 21.2 | yes |
| 16384 | 14.14 (13.87-14.54) | 29 | 25.3 | yes |
| all-GPU, no streaming | 28.04 (27.83-28.11) | n/a | 31.7 | reference |

Wired is the whole machine's peak, not a minimum RAM. Bitwise identical: the same logits in 6 of 6
checks and the same greedy text in 4 of 4. The bytes the disk read during decode were within 1% of the bytes direct
I/O requested. Expert pages and sidecar in the page cache: measured 0.0 % before and after each run.
