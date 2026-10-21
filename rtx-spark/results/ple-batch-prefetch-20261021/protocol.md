# Whole-batch PLE prefetch A/B protocol (written before measurement)

Hypothesis: issuing llama_prefetch_rows for the PLE rows of ubatches 2..4 at the start of each 2048-token decode call hides part of the measured lazy-PLE cold penalty (2K median 10.3%, 8K 8.5%).

Arms: control = llama.dll built from unmodified owned source (hash 0E564C7C...); candidate = same plus whole-batch prefetch (hash F1BD83B3...). Everything else is the installed C:\llama.cpp file set and the deployed recipe (-lzm on, -b 2048 -ub 512, MTP3 Q4_K draft head, dynamic vocab 16384, 262K ctx).

Order: A B B A (one server per phase). Every phase uses its own novel prompts (disjoint seeds, disjoint code files), 4 random-word + 2 code prompts at each of 2K and 8K. Each prompt runs cold then warm (cache_prompt=false).

Primary metric: per-length ratio of median cold prompt tok/s, candidate / control, pooled over the two phases of each arm.
Gate: ACCEPT if cold gain >= +3% at 2K and at 8K, warm gain >= -1% at both lengths, and a fixed 64-token greedy generation is identical between arms. REJECT if either cold gain < +1%. Otherwise INCONCLUSIVE (one replication allowed, then stop).
Upper bound note: the first ubatch of each decode call cannot benefit, so the ceiling is about 3/4 of the measured penalty.
