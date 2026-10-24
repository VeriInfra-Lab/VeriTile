import bench.tritonbench_g.kcache_copy_triton.Proofs.RawBitCopy
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
open VeriTile.Triton FP.Structural
open TypedCopy (Storage Parameters)
noncomputable section

/-- These five names have explicit raw payload widths. The old source's sixth
constructor, named fp64, is deliberately kept in the separate legacy branch. -/
inductive MachineFormat where
  | fp16 | bf16 | fp32 | f8e4 | f8e5
  deriving DecidableEq, Repr

def MachineFormat.width : MachineFormat → Nat
  | .fp16 | .bf16 => 16
  | .fp32 => 32
  | .f8e4 | .f8e5 => 8

def MachineFormat.storage : MachineFormat → Storage
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32 | .f8e4 => .f8e4 | .f8e5 => .f8e5

/-- Actual fp32 source reads use Cell.real; other machine source channels use
their named tags. The distinct RawCell.fp32 channel is preserved in memory but
is not silently mistaken for this source's fp32 representation. -/
def payload : (f : MachineFormat) → BitVec f.width → RawCell (BitVec 32)
  | .fp16, b => .fp16 b
  | .bf16, b => .bf16 b
  | .fp32, b => .real b
  | .f8e4, b => .f8e4 b
  | .f8e5, b => .f8e5 b

@[simp] theorem payload_storage (f : MachineFormat) (b : BitVec f.width) :
    readStorage f.storage (payload f b) = payload f b := by cases f <;> rfl

theorem payload_injective (f : MachineFormat) : Function.Injective (payload f) := by
  cases f <;> intro a b h <;> cases h <;> rfl

/-- Full accepted n_tokens=1/split-x source, for all five width-explicit raw
storage formats. The source's parameters, Nat metadata, aliases, zero width,
zero strides, arbitrary register seed and undef function are unrestricted.
The only function parameters are memory/launch state, not numerical callbacks.
This is word-addressed software storage, not native/GPU memory conformance. -/
specification five_format_raw_bit_correctness (f : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32))) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert) (TypedCopy.source f.storage p)
      (state m seed) = some final ∧
      final.mem=encodeMemory (reference f.storage p m seed.pids) ∧
      decodeMemory final.mem=reference f.storage p m seed.pids ∧
      (∀ i : TileIndex [p.width], final.mem p.KCache (outputOffset p m seed.pids i) =
        encode (readStorage f.storage (m p.K (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.KCache ∨ ∀ i : TileIndex [p.width], o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete f.storage p m seed

/-- Every payload, including signaling NaN bit patterns and both zero signs,
survives the complete source. This corollary states initial input contents;
it does not assume any final result. -/
theorem every_machine_payload (f : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [p.width]) (b : BitVec f.width)
    (hinput : m p.K (inputOffset p seed.pids i)=payload f b) :
    (TypedCopy.execution inert f.storage p (state m seed)).val.mem p.KCache
      (outputOffset p m seed.pids i) = encode (payload f b) := by
  simpa only [hinput,payload_storage] using execution_readback inert f.storage p m seed i

/-- The entire former abstract `.real` branch is retained, for arbitrary R.
There is no finite encoding assertion about arbitrary Structural.State α.
RawCell R has a complete representation, including the original default-on-
wrong-tag behavior, and the old unrestricted abstract theorem stays unchanged.
The source constructor's name fp64 does not prove a native64 storage channel. -/
theorem legacy_raw_carrier_correctness {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert) (TypedCopy.source .fp64 p)
      (state m seed) = some final ∧
      final.mem=encodeMemory (reference .fp64 p m seed.pids) ∧
      decodeMemory final.mem=reference .fp64 p m seed.pids ∧
      (∀ i : TileIndex [p.width], final.mem p.KCache (outputOffset p m seed.pids i) =
        encode (readStorage .fp64 (m p.K (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.KCache ∨ ∀ i : TileIndex [p.width], o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete .fp64 p m seed

/-- A 64-bit raw word is one exact legacy carrier instance. This proves no
native64 load lowering: the same erased source admits any R, including other
word widths. No truncation to the machine carrier's 32-bit packing occurs for
this branch; nondefault R values are stored in Carrier.legacy unchanged. -/
theorem legacy64_payload (p : Parameters) (m : Memory (BitVec 64))
    (seed : State (Carrier (BitVec 64))) (i : TileIndex [p.width]) (b : BitVec 64)
    (hinput : m p.K (inputOffset p seed.pids i)=RawCell.real b) :
    decode ((TypedCopy.execution inert .fp64 p (state m seed)).val.mem p.KCache
      (outputOffset p m seed.pids i)) = RawCell.real b := by
  rw [execution_readback,hinput]
  simp [readStorage]

/-- Explicit fp64 ComputeOp.load is unsupported by this evaluator. The legacy
source result above must therefore not be described as such a load theorem. -/
theorem explicit_fp64_load_not_interpreted {R : Type} [Inhabited R]
    (M : Algebra (Carrier R)) (mem : MemAccess .real shape) (mask : MaskOpt .real shape)
    (padding : PaddingOption) (s : State (Carrier R)) :
    FP.Structural.evalComputeOp M (ComputeOp.load .fp64 mem mask padding) s = none := rfl

#axiomsClean five_format_raw_bit_correctness
#axiomsClean payload_storage
#axiomsClean payload_injective
#axiomsClean every_machine_payload
#axiomsClean legacy_raw_carrier_correctness
#axiomsClean legacy64_payload
#axiomsClean explicit_fp64_load_not_interpreted
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
