| model                          |       size |     params | backend    | ngl | n_ubatch |  fa | lazy_mode  |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | ---------- | --------------: | -------------------: |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |          pp2048 |        903.55 Â± 6.62 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |            tg32 |         30.72 Â± 0.57 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         | pp2048 @ d16384 |       786.81 Â± 45.92 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |   tg32 @ d16384 |         29.52 Â± 0.75 |

build: 5e03a1ff4 (11681)

<!-- stderr tail -->
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 113128 MiB):
  Device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), compute capability 12.1, VMM: yes, VRAM: 113128 MiB
load_backend: loaded CUDA backend from C:\llama.cpp\ggml-cuda.dll
load_backend: loaded RPC backend from C:\llama.cpp\ggml-rpc.dll
load_backend: loaded CPU backend from C:\llama.cpp\ggml-cpu.dll