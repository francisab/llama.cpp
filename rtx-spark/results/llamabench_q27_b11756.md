| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |     2048 |   1 |          pp2048 |       962.09 Â± 11.74 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |     2048 |   1 |            tg32 |         15.59 Â± 0.13 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |     2048 |   1 | pp2048 @ d16384 |       817.26 Â± 29.24 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | CUDA       |  99 |     2048 |   1 |   tg32 @ d16384 |         15.00 Â± 0.20 |

build: d80619397 (11756)

<!-- stderr tail -->
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 113128 MiB):
  Device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), compute capability 12.1, VMM: yes, VRAM: 113128 MiB
load_backend: loaded CUDA backend from C:\llama.cpp\ggml-cuda.dll
load_backend: loaded RPC backend from C:\llama.cpp\ggml-rpc.dll
load_backend: loaded CPU backend from C:\llama.cpp\ggml-cpu.dll