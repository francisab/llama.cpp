| model                          |       size |     params | backend    | ngl | n_ubatch |  fa | lazy_mode  |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | ---------- | --------------: | -------------------: |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |          pp2048 |       887.60 Â± 21.55 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |            tg32 |         29.79 Â± 0.79 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         | pp2048 @ d16384 |       791.99 Â± 47.76 |
| qwen4exp A3B IQ4_XS - 4.25 bpw |  87.24 GiB |   176.94 B | CUDA       |  99 |     2048 |   1 | on         |   tg32 @ d16384 |         28.92 Â± 1.16 |

build: d80619397 (11756)

<!-- stderr tail -->
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 113128 MiB):
  Device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), compute capability 12.1, VMM: yes, VRAM: 113128 MiB
load_backend: loaded CUDA backend from C:\llama.cpp\ggml-cuda.dll
load_backend: loaded RPC backend from C:\llama.cpp\ggml-rpc.dll
load_backend: loaded CPU backend from C:\llama.cpp\ggml-cpu.dll