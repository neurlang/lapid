# lapid C API

`lapid.h` exposes the lapid engine to C and anything with a C FFI (C++, Swift, Go, C#, Dart,
Zig, Python ctypes, Kotlin/JNI): devices, `.lapid` model files, lazy tensors with one function
per op, and eval on CPU, CUDA, Vulkan or Metal.

## Download

Prebuilt archives are attached to the GitHub Releases tagged `c-v<version>`:

| Platform | Archive |
| --- | --- |
| Linux x86_64 / aarch64 | `lapid-c-<version>-{x86_64,aarch64}-unknown-linux-gnu.tar.gz` |
| macOS arm64 / x86_64 | `lapid-c-<version>-{aarch64,x86_64}-apple-darwin.tar.gz` |
| Windows x86_64 | `lapid-c-<version>-x86_64-pc-windows-msvc.zip` |

Each contains `include/lapid.h` and `lib/`: `liblapid.a` plus `liblapid.so` / `liblapid.dylib`
(Windows: `lapid.lib` static, `lapid.dll` with import library `lapid.dll.lib`), and
`native-static-libs.txt`, the system libraries a static link needs. Check downloads against
`SHA256SUMS`.

Build from source instead: `cargo build -p lapid-c --release` (outputs in `target/release`).

## Use

```c
#include "lapid.h"

lapid_check_abi(LAPID_ABI_VERSION);
lapid_device *dev;
lapid_device_open("cpu", &dev);            /* or "gpu", "cuda:0", "vulkan", "metal" */

const float a[] = {1, 2, 3, 4, 5, 6}, b[] = {1, 0, 0, 1, 1, 1};
const size_t as[] = {2, 3}, bs[] = {3, 2};
lapid_tensor *ta, *tb, *tc;
lapid_tensor_from_data(a, as, 2, LAPID_DTYPE_F32, &ta);
lapid_tensor_from_data(b, bs, 2, LAPID_DTYPE_F32, &tb);
if (lapid_matmul(ta, tb, &tc) != LAPID_OK)
    fprintf(stderr, "%s\n", lapid_last_error());  /* {"status":..,"error":..,"message":..} */

float out[4];
lapid_to_host_f32(dev, tc, out, 4);        /* runs the graph */
lapid_tensor_free(ta); lapid_tensor_free(tb); lapid_tensor_free(tc);
lapid_device_free(dev);
```

Every function returns a `lapid_status` (0 is `LAPID_OK`), handles are refcounted
(`*_retain` / `*_free`), and all functions are thread safe. The header documents ownership
for each call. Full examples: `examples/c/matmul.c` and `examples/c/model.c` (load a
`.lapid` file, read metadata and blobs, run a layer).

## Link

```sh
# shared
cc app.c -I include -L lib -llapid -Wl,-rpath,'$ORIGIN/lib' -o app      # Linux
cc app.c -I include -L lib -llapid -Wl,-rpath,@executable_path/lib -o app  # macOS

# static: add the system libraries from lib/native-static-libs.txt
cc app.c -I include lib/liblapid.a -ldl -lgcc_s -lutil -lrt -lpthread -lm -lc -o app   # Linux
cc app.c -I include lib/liblapid.a -liconv -framework Metal -framework CoreFoundation \
   -framework Foundation -lobjc -lc -lm -o app                                          # macOS
```

On Windows, link `lapid.dll.lib` and ship `lapid.dll`, or link `lapid.lib` plus the libraries in
`native-static-libs.txt`.

GPU drivers (CUDA, Vulkan, Metal) are loaded at runtime; without them lapid runs on the CPU.

## Develop

The header is generated with cbindgen and committed. After changing the API, run
`tools/check_c_header.sh --write` (needs `cargo install cbindgen --locked --version 0.29.2`);
CI-style check without `--write`. `cargo test -p lapid-c` compiles and runs the C examples
with the system C compiler when one is installed.
