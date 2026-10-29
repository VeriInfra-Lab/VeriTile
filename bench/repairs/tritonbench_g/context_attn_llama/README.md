# Checked direct page-gather context attention

The repaired Python preserves the original production kernel's floating AST.
Only program IDs, integer metadata/page loads and aranges are widened to int64.
The supplied output is updated directly; the wrapper returns `None`. CPU copies
are confined to integer metadata used by validation.

The contract is ordinary same-device CUDA FP32 Q/Out `[T,H,D]` and K/V
`[C,Hkv,D]`, with canonical contiguous strides, D16/32/64/128/256, positive
head counts and integral grouped heads. Start, total sequence length, prompt
length and request IDs are contiguous int32 vectors; the request-to-cache-row
table is canonical contiguous int32 `[R,S]`. Lazy negative/conjugate flags must
be resolved. Read roles may share storage. Output backing storage is disjoint
from every read role, tested by complete address intervals. Every nonempty
pointer is four-byte aligned; storage extents and all formed pointer expressions
meet the signed-int64 byte ABI.

Checked metadata satisfy `0 <= prompt <= sequence <= S`, valid request IDs,
valid cache rows for each selected sequence prefix, query capacity for the
uncached suffix, and disjoint positive-length packed output intervals. The
caller maximum is a nonnegative built-in signed-int32 integer covering every
uncached length. The launch uses their actual maximum, skips empty calls, and
preserves Tesla BM=BN64 versus BM=BN128, four/eight warps and one stage. The
owning-device context is restored to the caller on return.

`Checked.whole_mathematical_real` starts from arbitrary Real payload arrays.
It derives the original `Host.Valid` conditions from the checked finite input
contract and computes the independent gathered base-2 attention formula over
the whole grid, with complete logical coverage, all read regions, the complete
complement-of-writes frame and `None`. The finite `-1e8` causal sentinel and
padded denominator entries are preserved. It therefore does not state ideal
strict-causal softmax. The original host interface records the five FP32 scale
arguments produced by Python's ordered power/division/log2e multiplication.

`CheckedTensor.whole_tensor_fp32` binds the same original bytes and actual Tensor
metadata to source input leaves and output readback. The output word is an
independent FP recurrence in `FPReference`, with explicit page lookup, query/
key/value tile construction, initial state, online max/mass/accumulator updates,
probability casts and final division. Prelude, loop, epilogue and grid theorems
connect the actual typed source to that recurrence; noncircularity is audited.
The public result includes source/native grid success, exact output bytes,
logical Tensor values, output identity, read backing windows, untouched bytes,
empty calls, `None`, launch-device selection and caller-device restoration.

The selected numerical profile reuses the corrected shared SM90 fixed-k8 TF32
WGMMA expression algebra, ordered physical reductions, replica routes, FMA,
exp2 and division. The first QK instruction explicitly clears scale-d; PV
retains its live seed even when its bits are zero, and later instructions always
accumulate. Only eight scalar instruction families and a fixed-k8 single-output
WGMMA family remain local numerical ports. There is no whole-dot, reduction,
whole-output or native-to-Real equality premise. Historical generic/staging
modules remain reusable proof helpers; the new canonical call is direct.

Native facts derive from the same guard: actual typed source controls and
reached operands, byte events, signed all-lane formed offsets including masked
lanes, byte scaling, queried source strides, actual alignment, and complete
logical coverage. The widened original min/end and BN-step backedge fit int64
under the int32 metadata contract. Ordinary Tensor/storage/CPU observations,
b32 transport, compiler realization of the selected instruction profile, and
successful race-free launch delivery remain explicit boundaries. The shared
runtime launch-product issue remains tracked in `bench/TODO.md`.

`check_contract.py` proves exact floating-AST retention and executes the actual
repaired source in the CPU Triton interpreter with explicitly labeled CUDA
metadata/context facades. It passes all ten tile/width combinations, grouped
heads, gathered/duplicate cache pages, prompt caches, a multi-tile tail, packed
gaps and backing frames, empty calls, shared read storage, finite-sentinel
behavior, device restoration, invalid metadata/dtype/flags/shape, misalignment
remainders 1/2/3, and distinct physically overlapping storage objects. This is
CPU interpreter evidence, not GPU validation. Public proofs carry standard-base
axiom audits; original existing Python and Lean APIs remain unchanged.
