# vLLM Qwen 3.8 27B for Nvidia RTX 5090 (sm120)

> [!WARNING]
> This is both WiP and hacky. Don't use it on anything other than sm120.
>
> Don't attempt to modify vLLM launch parameters without checking against the patches.

- **Model** — [Minachist/Qwen3.8-27B-INT6-Mixed-AutoRound](https://huggingface.co/Minachist/Qwen3.8-27B-INT6-Mixed-AutoRound): 7.30bpw mixed INT5/6/7/8 of Qwen3.8 27B
- **Context** — 247,808
- **Layout** — 64 LM layers (48 GDN, 16 full attention), BF16 vision tower, MTP=3

## Patches

Applied by `patches/apply.sh` on vLLM `0.26.1rc1 (ba07e4a)` + FlashInfer `0.6.17`.

- **10-qwen-quantized-embeddings** — pass the quant config into `embed_tokens` of `qwen3_5.py` [quant]
- **20-qwen-mtp-quantized-embeddings** — same for the MTP module `qwen3_5_mtp.py` [quant]
- **30-flashinfer-shared-workspace** — shared FlashInfer workspace per device [memory]
- **40-triton-kv-scale-lifetime** — avoid int8 KV scale views bound to the temporary profiling KV tensor [quality]
- **50-flashinfer-int8-xqa** — FlashInfer XQA decode path for INT8 per-token-head KV cache [memory]
- **60-qwen-int8-hadamard** — Hadamard rotation for INT8 per-token-head KV with Qwen3.5 [quality]
- **70-host-mapped-embeddings** — keep input-embedding weights in pinned host RAM [memory]
- **80-gdn-direct-output** — GDN prefill scan writes directly into the preallocated output [memory]
- **90-mrope-cache-cap** — cap the MRoPE cos/sin cache at `max_model_len` [memory]
- **100-gdn-state-replay** — replace speculative GDN state blocks with state tape plus replay kernel [memory]
- **110-async-accepted-counts** — fix accepted-token counts under async scheduling [bugfix]
- **120-startup-memory-check-bypass** — warn instead of fail on startup free-memory check [memory]
- **130-vision-weight-swap** — swap decoder pages for vision encoding [memory]

## Evaluation

> [!NOTE]
> Will follow.
