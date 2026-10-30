# Scalar CUDA load/store audit

Reviewed 2026-10-19 for `add_example`, FP32, `BLOCK_SIZE = 4`, Triton 3.7.1.
The 4-byte alignment condition is justified for the emitted scalar CUDA memory
instructions. It is not an assumption inferred merely from FP32 element size
or Tensor contiguity. No change to the alignment guard or memory predicates is
needed on the evidence below.

## Normative requirement

The shared alignment, mask and transfer rules and their normative references
live in [GpuMemoryModel.md: Physical byte-address contracts](../../../documents/GpuMemoryModel.md#physical-byte-address-contracts).
In particular, [PTX ISA 8.7 section 6.4.1](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#addresses-as-operands)
requires alignment to the instruction's total access size. Applied to the
scalar `.b32` lowering observed below, this requires 4-byte alignment; it does
not establish the alignment needed by arbitrary vector instructions.

The installed 3.7.1 `triton/language/core.py` and `semantic.py` were also checked:
false mask elements suppress accesses, pointer element types determine load
types, and store values convert to the pointer element type. The rest of this
document records this task's model binding and compiler evidence.

## Observed compilation

The original and revised `add_kernel` ASTs are identical. Offline compilation
used the installed NVIDIA backend, default launch options, explicit SM80 and
SM90 targets, and three representative specialization configurations: no
alignment hint, all pointers divisible by 16, and pointers plus extent divisible
by 16. All six variants emitted PTX 8.7 with two scalar `ld.global.b32`
instructions and one scalar `st.global.b32` instruction:

```ptx
setp.lt.s32  %p1, %r9, %r6;
mul.wide.s32 %rd7, %r9, 4;
add.s64     %rd1, %rd4, %rd7;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
// The second input uses the same offset and mask.
// The store predicate combines the source mask with duplicate-lane suppression.
@%p2 st.global.b32 [ %rd3 + 0 ], { %r3 };
```

The no-hint variant still uses `.b32`; it does not fall back to byte loads or
assemble an unaligned FP32 value from four bytes. The parameter's `.align 1`
annotation in this variant does not override the memory instruction's PTX
alignment requirement. Runtime `BaseBackend.get_tensor_specialization` checks
divisibility by 16 for an optimization hint. It does not reject pointers with
remainders 1, 2 or 3 modulo 4.

## Comparison with the proof

| Property | Model and evidence |
| --- | --- |
| Pointer arithmetic | `PhysicalView.nativeAt` sign-extends the 32-bit element offset and multiplies by the scalar byte width in 64-bit arithmetic. LLVM IR uses `sext` and `getelementptr float`; PTX uses `mul.wide.s32 ..., 4` and `add.s64`. |
| Mask and integer bounds | The model uses signed 32-bit comparison, matching `setp.lt.s32`. The checked extent and existing index proof prevent signed overflow for launched lanes. |
| Inactive lanes | `executeInstruction` performs no load/store for a false mask. Its arbitrary inactive load value cannot affect an active output in this pointwise kernel, as `source_program_reduction` proves. Emitted memory instructions are predicated. |
| Active access validity | `ReadOK`/`WriteOK` require scalar natural alignment, in-bounds byte spans, nonwrapping addresses and device permissions. `Native.launch_correct` proves these properties for every active source access. |
| Values and bytes | Emitted loads/stores transport 32 bits; LLVM bitcasts connect those bits to FP32 arithmetic without a storage-width conversion. `MemoryPrimitives` exposes byte packing and exterior-frame laws; these remain ordinary trusted memory laws. |
| Aliasing and output writes | The inputs may overlap. The fresh output, distinct output cells and existing frame proofs prevent stores from changing input values or unrelated bytes. |

`ReadOK` and `WriteOK` are applicability predicates, not an interpreter for GPU
faults. The model's total mathematical memory functions do not claim defined
native behavior outside those predicates. The upper `VeriTile.Triton.Memory`
named-cell abstraction likewise does not by itself certify physical access
validity; this task supplies the separate finite-address bridge.

## Reference map and trust boundary

The relevant definitions now carry citations next to their contracts. References
support the stated external semantics; they do not replace the Lean proofs or
turn trusted API/allocator laws into verified implementations.

| Definition or premise | Reference and status |
| --- | --- |
| `ReadOK`, `WriteOK`, `InputContract.alignment` | PTX ISA 8.7 section 6.4.1 above supplies the normative access-size alignment rule. The six offline cases establish that this kernel's checked variants use scalar `.b32`. |
| `MemoryPrimitives.read` / `write` | [PTX ld](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld) and [PTX st](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st) specify memory transfers. Concrete packing, allocation permissions and exterior-frame laws remain explicit trusted interface contracts. |
| `PhysicalView.nativeAt` | PTX section 6.4.1 specifies byte addresses. The signed element-offset extension and scaling are observed compiler results recorded above; they are not universal properties of every Triton index type. |
| `executeInstruction` masks | The Triton load/store references in the shared memory document specify suppressed accesses. [PTX Predicated Execution](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#predicated-execution) describes the instruction guards seen in the output. |
| `TensorLaws.pointer` / `.contiguous` | PyTorch [data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html), [storage_offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html) and [is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html) document the first-element address, offset in elements, and memory-format query. The corresponding installed Torch 2.13.0 docstrings were checked. `TensorLaws` remains a snapshot binding to the resolved pointer and storage, not a proof of Torch implementation or host-query purity. |
| `logicalValue` and `InputContract.noNeg` | [torch.resolve_neg](https://docs.pytorch.org/docs/stable/generated/torch.resolve_neg.html) distinguishes the negative view bit from materialized negation. The checked absence of that bit connects direct storage reads to logical input values. |
| `ZerosLikeResult` | [torch.zeros_like](https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html) specifies zeros and inherited metadata, with `preserve_format` by default. Freshness, output alignment, pointer binding and the byte frame remain explicit trusted allocator contracts; the API page alone does not prove them. |

The installed `data_ptr` documentation notes that observing a copy-on-write
tensor's pointer may materialize its copy. The binding above describes resolved
launch pointers; its citation must not be read as a guarantee that all Tensor
pointer queries are pure. Torch query/allocator internals remain at the stated
API boundary, separate from the proven kernel execution.

## Scope and reproduction

```sh
python bench/tritonbench_g/add_example/check_cuda_lowering.py \
  --output-dir /tmp/veritile-add-example-load-store-audit
```

The checker saves TTIR, TTGIR, LLVM IR, PTX and a versioned report. It checks
the original kernel AST, scalar access width, memory predication and signed
offset scaling. This is compiler inspection and a regression check, not a
formal compiler-correctness proof or exhaustive specialization coverage.

No contradiction was found in the scalar FP32 load/store bridge used by this
task. This does not certify every Triton memory operation: vectorized accesses
need instruction-width alignment and bounds justified by their lowering;
other backends, atomics, block pointers and asynchronous copies need their own
applicable semantics. No broader wrapper restriction is introduced here.

CUDA execution was unavailable. There was no GPU run, GPU misalignment probe,
or numerical conformance experiment. A successful misaligned GPU run would not
establish defined behavior under this specification. Native-to-IEEE primitive
conformance is a modeling boundary, not a deferred transformation experiment.
The current kernel transformation uses the existing accepted FP32 ADD-COMMUTE
report and requires finite operands only on active observed lanes. The original benchmark
is not certified for arbitrary Python-accepted inputs by this audit.
