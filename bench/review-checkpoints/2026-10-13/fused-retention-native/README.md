# Unchanged retention JITs: offline native compilation

The complete 864-case matrix finishes with unchanged captured inputs on
Triton 3.8.0 / sm90. Forward has 288 compiled and 288 rejected configurations;
backward has 144 compiled and 144 rejected configurations. All 2,160 retained
TTIR, TTGIR, LLVM IR, PTX and cubin artifact hashes have been checked after
archiving. The cache is omitted; the independently saved per-case artifacts,
errors, exact extracted source, runner and toolchain fingerprints are retained.
Receipt paths refer to the original run directory; each case has the same
relative path in this archive.

All tested fp16/bf16/fp32/fp64 input profiles compile at the three valid
wrapper block shapes, for every tested output dtype and seed/final flag.
Both fp8 input formats are rejected at the original masked load's integer
zero conversion. The original DK/DV=3/5 wrapper choice retains non-power-of-two
arange dimensions and is rejected; no block-size repair is substituted.

Only 144 cases obey the wrapper's output-dtype-equals-input rule: 72 compile
and 72 are rejected. The other storage pairs are direct-JIT observations.
B/H/T/strides remain runtime signed i32 values and scale is runtime fp32;
documented T examples are neither compile-time specializations nor executions.
No kernel is launched. Host reductions, fixed-width overflow, native numerical
correctness, arbitrary shape admission and independent proof replay are open.
