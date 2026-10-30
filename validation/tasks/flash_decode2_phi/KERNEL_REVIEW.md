# Kernel-only review: admitted width exceeds the Triton arange domain

The selected source remains `flash_decode2_phi_repaired.py`, SHA256
`6eec28122238f0f4ea7352b8b8f52ab2a37aae7823c828c974ba392c465d61e1`.
This review leaves it and all 25 pre-existing task sources/proofs unchanged.

The current wrapper permits every positive dimension whose next power of two
is at most `2^30`, subject to the existing shape, storage, dtype, length and
address checks. Triton 3.7.1's documented `tl.arange` API permits at most
`TRITON_MAX_TENSOR_NUMEL = 1048576` elements. Its frontend `validate_block_shape`
enforces this bound before backend optimization, register allocation or GPU
execution. The formal Geometry.Valid lane bound of `2^30` therefore does not
establish source compilability for the entire admitted domain.

A concrete admitted call has `B=H=S=Length=block_seq=1`, `D=1048577`, Mid shape
`[1,1,1,1048577]`, Log shape `[1,1,1]`, Length shape `[1]`, and Out shape
`[1,1,1048577]`. Mid and Log are zero FP32, Length is int32 one, and Out is FP32
seven. All buffers are ordinary contiguous zero-offset views with distinct
storage and no lazy flags. The actual unmodified wrapper accepts them under a
CPU Tensor CUDA-admission facade and captures grid `(1,1)` with
`BLOCK_DMODEL=2097152`. All scalar arguments fit int32; signed64/uint64 span
checks pass. The grid product is one.

Compiling the actual selected JIT with its captured lane/block constants,
`num_warps=4`, `num_stages=2` and explicit CUDA SM80 target produces
`CompilationError` at `offs_d = tl.arange(0, BLOCK_DMODEL)`:

```
numel (2097152) exceeds triton maximum tensor numel (1048576)
```

The same offline compilation setup succeeds for the control `D=3`,
`BLOCK_DMODEL=4`, and produces saved load/store PTX. The failure is a frontend
API-domain rejection, independent of numerical payloads and backend resource
availability. There is no GPU execution or observed live CUDA-launch error.
The CPU facade establishes actual Python admission and captured arguments; the
separate actual-JIT compiler run establishes this deterministic frontend error.

The independent mathematical stage-2 target is retained:
`sum(j<K, exp(Log[b,h,j])*Mid[b,h,j,d]) / sum(j<K, exp(Log[b,h,j]))`, with
`K=ceil(Length[b]/block_seq)`. It describes supplied summaries and assumes no
stage-1/raw-token identity. The source's online maximum/rescaling loop, Phi's
`d<D` masks, complete stores, native primitive operation order and existing
Tensor/address/frame mathematics remain preserved. Those conditional internal
results do not remove the admitted source-domain counterexample. Primitive
scalar/exp/FMA interpretation is a model boundary; this review creates no new
numerical-experiment obligation.

A repair would need an explicitly reviewed input-domain or kernel design
change. This review introduces neither a hidden width restriction nor a new
split kernel, and does not claim that merely applying the arange maximum would
prove every remaining backend resource condition. Root owns repair selection,
independent review, Lake and task-ledger classification.

Reproducible packet: `.veritile-cache/flash-decode2-phi-kernel-evidence/` contains
`launch_compile_probe.py`, `probe-report.json`, the exact compiler error, the
successful control PTX, installed API/compiler-source snapshots, the selected
ledger snapshot and all pre-existing source hashes. No existing proof or shared
Phi helper is modified, including helpers reused by `flash_decode2_llama`.
