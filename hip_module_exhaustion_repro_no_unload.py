import argparse
import ctypes
import os
import pathlib
import tempfile


def compile_minimal_hsaco(cache_dir: str) -> pathlib.Path:
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device not available.")

    @triton.jit
    def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(z_ptr + offsets, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    add_kernel[grid](x, y, z, n, BLOCK=256)
    torch.cuda.synchronize()

    hsacos = list(pathlib.Path(cache_dir).rglob("*.hsaco"))
    if not hsacos:
        raise RuntimeError(f"No .hsaco found in {cache_dir}")
    return max(hsacos, key=lambda p: p.stat().st_mtime)


def load_until_fail(hsaco_path: pathlib.Path, max_loads: int, log_every: int) -> int:
    data = hsaco_path.read_bytes()
    buf = ctypes.create_string_buffer(data)
    image_ptr = ctypes.c_void_p(ctypes.addressof(buf))

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipSetDevice.restype = ctypes.c_int
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    hip.hipModuleLoadData.restype = ctypes.c_int

    hip.hipSetDevice(0)

    count = 0
    while count < max_loads:
        module = ctypes.c_void_p()
        res = hip.hipModuleLoadData(ctypes.byref(module), image_ptr)
        if res != 0:
            print(f"[repro] load_failed res={res} count={count}")
            return res
        count += 1
        if count % log_every == 0:
            print(f"[repro] loaded={count}")

    print(f"[repro] reached max_loads={max_loads} without error")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-loads", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--cache-dir", type=str, default="")
    args = parser.parse_args()

    if args.cache_dir:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = tempfile.mkdtemp(prefix="triton_cache_")

    hsaco = compile_minimal_hsaco(cache_dir)
    print(f"[repro] using hsaco: {hsaco}")
    return load_until_fail(hsaco, args.max_loads, args.log_every)


if __name__ == "__main__":
    raise SystemExit(main())

