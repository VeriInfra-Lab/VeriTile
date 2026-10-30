# matmul_triton1: checked contiguous FP32 matrices

The original benchmark is unchanged. This revision supplies an explicit input
contract, legal fixed-size tiles, complete tail masks and finite-width indexing.
The mathematical and source-precision results cover the checked wrapper,
allocation, empty return, complete launch and actual returned modeled Tensor.

## Supported contract and repair

`matmul_triton1_repaired.py` accepts two strided, rank-two FP32 CUDA tensors
on `cuda:0`, with compatible matrix dimensions. Both are contiguous and have
resolved negative view bits. The Torch default dtype must be FP32 because the
original output allocation has no explicit dtype. Each of M, N and K is less
than `2**63`. Each nonempty input has a data pointer divisible by four; empty
inputs impose no alignment requirement and their pointers are not queried.

The wrapper passes the original tensors directly. Contiguous offset views,
overlapping inputs, singleton dimensions with unusual strides, empty matrices,
K=0, NaNs and infinities are admitted. Canonical dense strides `K,1,N,1,N,1`
describe the accessed elements even when an irrelevant singleton stride is
unusual. No input copy, arbitrary-stride extension or finite-value guard is
introduced. Allocation is fresh on `cuda:0`; empty M or N returns it without
launching. The launch runs in the input device context.

The original kernel assumes physical row-major inputs without checking them.
For `x = arange(256).reshape(16,16).T` and an identity right operand, logical
output `[0,1]` is 16 while the original physical access reads 1. Its whole-shape
tile also fails to provide legal tile sizes for general matrices. The repair
uses 32-by-32-by-32 tiles, masks M/N/K loads and output stores, and retains
`allow_tf32=False`. Runtime dimensions/strides, program ID and lanes are widened
to uint64 before arithmetic. This also repairs signed-int32 offset overflow:
row 2 times K=`2**30` already reaches `2**31`.

## Proof entries

All names below are in `VeriTile.Bench.Repairs.MatmulTriton1.Revision001`.

| Entry | Strong public result |
| --- | --- |
| `Correct.lean` | `matmul_triton1_checked_real`: independent full-K mathematical sum of the original Tensor logical values. |
| `FPEquiv.lean` | `matmul_triton1_checked_native`: exact padded, ordered source computation using local FP32 instruction ports. |
| `FPEquiv.lean` | `matmul_triton1_checked_fp32_if_local`: transport to the FP32 reference under explicit local zero/FMA correspondence. |

`Contract.lean` names the supported domain independently of guard order.
`Proofs/Contract.lean` proves both `checkInputs_iff` and
`checkTensorInputs_iff`: admission equals that domain, including non-Tensor
rejection before attribute access. The shared contract also serves matmul2.

`Kernels.lean` contains source definitions and imports no proof module. The
real and FP entries independently import helpers; neither imports the other.
Existing logical/grid/host results remain available with their original scope.
The new proof chain is:

- `FiniteIndex` proves actual ABI casts, uint64 block/lane arithmetic, signed
  loop bounds, masks, pointer recurrence, active four-byte address windows and
  finite byte-pointer addition. Allocation capacity supplies ordinary storage
  bounds; no signed-int32 input-shape cap is assumed.
- `FiniteExecution` executes finite tests, loads, local dot, loop increments and
  stores. `FiniteSource` connects this executor to the complete typed source
  and the existing grid coverage/unique-ownership proof.
- `TensorBinding` reuses `VeriTile.Torch.FP32` to bind the original data pointer,
  checked alignment, storage offset and logical values to a common physical
  reader. Inputs may share storage. `InputAdapter` derives canonical descriptors
  from admission.
- `FiniteHost` and `CheckedHost` compose this execution with guards, allocation,
  empty return and exact returned allocation. `LogicalInputs` independently
  defines the mathematical sum and precision-specific padded fold.

The output postcondition includes rank, shape, FP32 dtype, device, contiguity,
resolved negative flag, fresh object identity, every output value and the frame
outside the fresh output. `CheckedHost.Allocation` describes successful
allocation for this actual call, including capacity; it does not assert that
every arbitrary shape can allocate. `Launchable` supplies a representable
successful native grid (`grid < 2**31`), separately from input admission.
Ordinary Tensor/storage laws, fresh allocation and successful race-free launch
remain runtime boundaries. The shared large-grid launcher work is deferred as
recorded in `bench/TODO.md`; no affected unvalidated runtime is certified here.

## Precision and validation boundary

The selected Triton 3.7.1 SM90 lowering uses scalar `fma.rn.f32` and keeps the
live accumulator in `tt.dot`; no TF32 operation occurs. `Native.dot` follows
ascending K lanes, including masked zero padding and the live carry. The
structural theorem retains the actual local FMA/zero ports.

`Native.Correspondence` requires zero and one scalar FMA to match
`FP.Config.ieeeValue`. Its domain is all FP32 words and all seeds, including
NaNs, infinities, signed zeros and subnormals; finite inputs alone would not
discharge it. The conditional theorem composes this local premise through the
whole wrapper. No whole-dot, tile or output-equality premise is used. GPU
instruction conformance, including exceptional-value and FTZ behavior, remains
unvalidated. The Real theorem interprets arithmetic over reals and is separate
from FP-to-Real equality.

Targeted Lean 4.29.0 builds and `#axiomsClean` checks cover both current matmul
dependency closures, including retained reader entries. There is no `sorry` or
new axiom declaration. Run `python check_contract.py 1` from this directory for
the focused source/host check: 27 pre-allocation rejection cases and 16
CPU-interpreted cases passed. It checks the kernel AST, guards, finite casts,
masks, original failure witnesses, tails, K=0, empty/offset/overlapping views,
singleton strides, NaN/Inf admission and exact returned allocation. CUDA launch
and allocation are mocked in these tests.

Offline compilation of this one configuration with i32 and i64 runtime
metadata passed on SM90. The shared run also compiled all eight matmul2
configurations, for 18 cases total. These are compiler inspections, not GPU
executions or a compiler-correctness proof. Torch was 2.13.0+cu130, Triton 3.7.1;
CUDA execution was unavailable.
