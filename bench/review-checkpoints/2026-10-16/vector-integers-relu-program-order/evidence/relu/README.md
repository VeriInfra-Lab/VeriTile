# Original ReLU scope extension

This frozen package adds six Lean modules for the actual original
`relu_triton_kernel` behavior. Only program zero stores. The full ascending
grid therefore fills at most the first block and retains the allocation's
arbitrary prior values everywhere else, including index 1024 in the wrapper.

The public `relu_typed_grid_correctness` is a **backend value contract** over
twelve logical input profiles: bool, eight signed/unsigned integer labels,
fp16, bf16 and fp32. Integer cells contain decoded Int/Nat/Bool values; the
width labels do not establish physical decoding or finite-width native
address arithmetic. The public comparison, literal and conversion callbacks
remain open. A separate concrete fp32 bit-order comparison/select helper
proves exact software word behavior, including NaNs and negative zero.

`HostWrapper` preserves fallible device access before allocation,
`empty_like(..., dtype=float32, device=x.device)`, output-derived numel, the
unused num_blocks computation, launch, return, all four literal harness
cases, dictionary insertion order and `result_gold` assignment. Its successful
logical launch is composed with actual typed-grid execution. Native metadata,
allocation, literal construction, launch, Python/dictionary/import behavior
remain explicit boundaries; successful harness key shape is not native
numerical harness completion.

The exact-source offline audit covers 17 representative canonical input
signatures with fp32 output at sm90. Thirteen produce frontend IR. The two
ordinary fp8 formats fail comparison with the integer-zero literal; two fnuz
formats fail architecture admission. The successful fp64 profile is not in
the twelve logical execution profiles. Shell/complex/quantized dtype domains
and all full native errors remain structural debt. No GPU run or native
backend compilation is claimed.

Whole-original-program acceptance remains false. Numerical followups in
`evidence/scope-review.json` are descriptive; root integration must bind any
future acceptance queue entry to exact declarations and source scope.

Reproduce from the repository cwd with the established toolchain:

```
python /tmp/veritile-standardization-20261015/relu-original-wip/freeze.py before
/tmp/veritile-lean-toolchain/bin/lake env python3 -B /tmp/veritile-standardization-20261015/relu-original-wip/verify.py bench/tritonbench_g/relu_triton_kernel/Proofs/GatedGrid.lean bench/tritonbench_g/relu_triton_kernel/Proofs/TypedValue.lean bench/tritonbench_g/relu_triton_kernel/Proofs/TypedGrid.lean bench/tritonbench_g/relu_triton_kernel/Proofs/HostWrapper.lean bench/tritonbench_g/relu_triton_kernel/Proofs/SoftwareFP32.lean bench/tritonbench_g/relu_triton_kernel/TypedGridCorrect.lean
python /tmp/veritile-standardization-20261015/relu-original-wip/freeze.py after
```

The isolated directory has only two dependency `.olean` symlinks to the
existing repository build (`Kernels` and `Proofs/FP`); no repository or shared
toolchain files are changed. Dependency freezes include actual
`lakefile.toml`, `lake-manifest.json`, `lean-toolchain` and the original Python.
