#!/usr/bin/env python3
"""Build an expert-major sidecar of the routed experts of a GGUF shard.

For each layer (in blk order) and each expert e, the block of e is
gate[e] | up[e] | down[e] (the parts in the MoE cache's bank order:
gate_up, gate, up, down), each part padded with zeros to ALIGN bytes.
Blocks of a layer are contiguous, so a run of consecutive experts is one
contiguous range; each layer starts at an ALIGN-aligned offset.

Writes <shard>.experts-major.bin, verifies it by re-reading it without the
page cache against the sha256 of every source tensor, and only then writes
<shard>.experts-major.idx (via a temporary name). Exits non-zero on any
mismatch. Tested on macOS, where reads and writes bypass the page cache
(F_NOCACHE). On Linux, which is untested, every chunk read or written is
dropped from the page cache instead (fdatasync, then posix_fadvise
DONTNEED), and the whole sidecar is dropped before it is verified.

usage: python3 -I scripts/build-moe-sidecar.py [--gguf-py <dir>] [--headroom-gib N] <shard.gguf>

--gguf-py defaults to the gguf-py directory of this repository.
"""

import argparse
import fcntl
import hashlib
import os
import re
import sys
import types
import time

ALIGN = 16384
FORMAT = 1
PART_ORDER = ("ffn_gate_up_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
CHUNK = 1 << 30
DARWIN = sys.platform == "darwin"
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)


def pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def open_nocache(path: str, flags: int) -> int:
    fd = os.open(path, flags, 0o644)
    if DARWIN:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def drop_cache(fd: int, offs: int, n: int, written: bool = False) -> None:
    """Linux: drop [offs, offs + n) of fd from the page cache; written data is synced first (dirty pages stay)."""
    if DARWIN:
        return
    if written:
        os.fdatasync(fd)
    os.posix_fadvise(fd, offs, n, os.POSIX_FADV_DONTNEED)


def pread_exact(fd: int, buf: memoryview, offs: int) -> None:
    done = 0
    while done < len(buf):
        n = os.preadv(fd, [buf[done:done + CHUNK]], offs + done)
        if n <= 0:
            raise OSError(f"short read at offset {offs + done}")
        done += n


def write_exact(fd: int, buf: memoryview, offs: int) -> None:
    done = 0
    while done < len(buf):
        n = os.pwrite(fd, buf[done:done + CHUNK], offs + done)
        if n <= 0:
            raise OSError(f"short write at offset {offs + done}")
        done += n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-py", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "gguf-py"),
                    help="the gguf-py directory (default: the one of this repository)")
    ap.add_argument("--headroom-gib", type=int, default=2, help="free disk to keep after the sidecar (default 2)")
    ap.add_argument("shard")
    args = ap.parse_args()
    if not DARWIN and not sys.platform.startswith("linux"):
        print(f"not supported on {sys.platform} (tested on macOS)", file=sys.stderr)
        return 2

    sys.path.insert(0, args.gguf_py)
    try:
        import yaml  # noqa: F401
    except ImportError:  # gguf-py imports it for model cards; the reader does not use it
        sys.modules["yaml"] = types.ModuleType("yaml")
    from gguf import GGUFReader  # noqa: E402

    shard = os.path.abspath(args.shard)
    shard_size = os.stat(shard).st_size
    out_bin = shard + ".experts-major.bin"
    out_idx = shard + ".experts-major.idx"
    for p in (out_bin, out_idx):
        if os.path.exists(p):
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2

    # metadata only, the mapped data is never touched
    reader = GGUFReader(shard)
    pat = re.compile(r"^blk\.(\d+)\.(" + "|".join(PART_ORDER) + r")\.weight$")
    layers: dict[int, dict[str, tuple[int, int, int]]] = {}
    for t in reader.tensors:
        m = pat.match(t.name)
        if not m:
            continue
        dims = [int(d) for d in t.shape.tolist()]
        if len(dims) != 3:
            raise SystemExit(f"{t.name}: expected 3 dims, got {dims}")
        n_expert = dims[2]
        if t.n_bytes % n_expert != 0:
            raise SystemExit(f"{t.name}: {t.n_bytes} bytes not divisible by {n_expert} experts")
        layers.setdefault(int(m.group(1)), {})[m.group(2)] = (int(t.data_offset), int(t.n_bytes), n_expert)
    del reader
    if not layers:
        print(f"no expert tensors in {shard}", file=sys.stderr)
        return 2

    # layout
    plan = []  # (il, base, block, n_expert, [(name, src_offs, nb2, part_offs)])
    base = 0
    for il in sorted(layers):
        parts_in = layers[il]
        n_experts = {v[2] for v in parts_in.values()}
        if len(n_experts) != 1:
            raise SystemExit(f"blk.{il}: expert tensors disagree on n_expert {n_experts}")
        n_expert = n_experts.pop()
        parts = []
        po = 0
        for kind in PART_ORDER:
            if kind not in parts_in:
                continue
            src_offs, n_bytes, _ = parts_in[kind]
            nb2 = n_bytes // n_expert
            parts.append((f"blk.{il}.{kind}.weight", src_offs, nb2, po))
            po += pad(nb2)
        block = po
        plan.append((il, base, block, n_expert, parts))
        base = pad(base + block * n_expert)
    total = base

    st = os.statvfs(os.path.dirname(out_bin))
    free = st.f_bavail * st.f_frsize
    if free - total < (args.headroom_gib << 30):
        print(f"not enough disk: {total} bytes needed, {free} free, {args.headroom_gib} GiB headroom kept", file=sys.stderr)
        return 2

    t0 = time.time()
    shas: dict[str, str] = {}
    src = open_nocache(shard, os.O_RDONLY)
    dst = open_nocache(out_bin, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        for il, lbase, block, n_expert, parts in plan:
            out = bytearray(block * n_expert)  # zero padding
            mo = memoryview(out)
            for name, src_offs, nb2, po in parts:
                buf = bytearray(nb2 * n_expert)
                mb = memoryview(buf)
                pread_exact(src, mb, src_offs)
                drop_cache(src, src_offs, len(mb))
                shas[name] = hashlib.sha256(mb).hexdigest()
                for e in range(n_expert):
                    o = e * block + po
                    mo[o:o + nb2] = mb[e * nb2:(e + 1) * nb2]
                del mb, buf
            write_exact(dst, mo, lbase)
            drop_cache(dst, lbase, len(mo), written=True)
            del mo, out
            print(f"wrote blk.{il}: {n_expert} experts, block {block}, at {lbase} ({time.time() - t0:.1f} s)", flush=True)
        os.ftruncate(dst, total)
        os.fsync(dst)
        if DARWIN:
            fcntl.fcntl(dst, F_FULLFSYNC)
        drop_cache(dst, 0, 0)  # 0, 0: the whole file
    finally:
        os.close(src)
        os.close(dst)
    t_write = time.time() - t0

    # verify: re-read without the page cache, every part against the sha256 of its source tensor, padding zero
    t1 = time.time()
    if os.stat(out_bin).st_size != total:
        print(f"FAIL: {out_bin} has {os.stat(out_bin).st_size} bytes, expected {total}", file=sys.stderr)
        return 1
    bad = 0
    fd = open_nocache(out_bin, os.O_RDONLY)
    try:
        for il, lbase, block, n_expert, parts in plan:
            buf = bytearray(block * n_expert)
            mb = memoryview(buf)
            pread_exact(fd, mb, lbase)
            drop_cache(fd, lbase, len(mb))
            hs = {name: hashlib.sha256() for name, _, _, _ in parts}
            pad_ranges = [(po + nb2, po + pad(nb2)) for _, _, nb2, po in parts if nb2 % ALIGN != 0]
            zeros = bytes(ALIGN)
            for e in range(n_expert):
                b = e * block
                for name, _, nb2, po in parts:
                    hs[name].update(mb[b + po:b + po + nb2])
                for a, z in pad_ranges:
                    if mb[b + a:b + z] != zeros[:z - a]:
                        print(f"FAIL: blk.{il} expert {e}: nonzero padding", file=sys.stderr)
                        bad += 1
            for name, h in hs.items():
                if h.hexdigest() != shas[name]:
                    print(f"FAIL: {name}: sidecar sha256 {h.hexdigest()} != source {shas[name]}", file=sys.stderr)
                    bad += 1
            del mb, buf
        tail_end = os.stat(out_bin).st_size
        last_end = plan[-1][1] + plan[-1][2] * plan[-1][3]
        if tail_end > last_end:
            tail = bytearray(tail_end - last_end)
            pread_exact(fd, memoryview(tail), last_end)
            if tail.count(0) != len(tail):
                print("FAIL: nonzero tail", file=sys.stderr)
                bad += 1
    finally:
        os.close(fd)
    t_verify = time.time() - t1
    if bad:
        print(f"FAIL: {bad} mismatches, no index written", file=sys.stderr)
        return 1

    lines = [
        f"moe-experts-major {FORMAT}",
        f"alignment {ALIGN}",
        f"source {os.path.basename(shard)} {shard_size}",
        f"bin {os.path.basename(out_bin)} {total}",
        "# tensor <name> <n_expert> <nb2> <layer_base> <block_size> <part_offs> <source_offs> <sha256 of the expert bytes in source order>",
    ]
    for il, lbase, block, n_expert, parts in plan:
        for name, src_offs, nb2, po in parts:
            lines.append(f"tensor {name} {n_expert} {nb2} {lbase} {block} {po} {src_offs} {shas[name]}")
    lines.append("end")
    tmp = out_idx + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, out_idx)

    print(f"OK: {len(shas)} tensors in {len(plan)} layers, bin {total} bytes ({total / 2**30:.2f} GiB), "
          f"source {shard_size} bytes; write {t_write:.1f} s, verify {t_verify:.1f} s")
    print(f"bin {out_bin}\nidx {out_idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
