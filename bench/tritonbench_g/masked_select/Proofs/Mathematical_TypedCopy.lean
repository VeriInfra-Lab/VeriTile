import bench.tritonbench_g.masked_select.Proofs.Mathematical_MaskedSelect
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

/-! Actual typed selection-copy source. Prefix sums and selection masks are
inputs to this stage; no prefix-sum producer or expected output is assumed.
The floating carrier is opaque and the integer payload channel is exact Int.
No arithmetic is performed on payloads. -/
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open Lean VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def inputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.width + i.1.val

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  (s.mem p.prefixSum.cast (inputOffset p s i)).read .nat - 1

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Prop :=
  (s.mem p.selection.cast (inputOffset p s i)).read .bool = Bool.true ∧ inputOffset p s i < p.size
instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if active p s i then t.write p.output (outputOffset p s i) (loaded d p s i) else t) s

def execution {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source d p) s = some t} := by
  cases d <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source,VeriTile.Bench.Repairs.MaskedSelect.kernel, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
       FP.Structural.bop, FP.Structural.natLt, copyBackend, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
     try rfl)

private def payloadDType : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | .store dtype _ _ _ _ => some dtype
  | _ => none

/-- The int64 source load/store use the integer payload channel, rather than
renaming a mathematical real load/store as an integer implementation. -/
theorem int64_payload_dtype (p : Parameters) :
    (source .int64 p).surfaceBody[3]?.bind payloadDType = some .int ∧
      (source .int64 p).surfaceBody[6]?.bind payloadDType = some .int := ⟨rfl, rfl⟩

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d
  all_goals simp +contextual [execution, copied, masked_scatter_memory, loaded,
    Storage.tag, active, inputOffset, outputOffset, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, and_assoc, and_left_comm, and_comm]

/-- Address collisions are excluded only among selected in-bounds lanes.
Unselected prefix entries may repeat freely. -/
theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) =
      .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag) := by
  rw [execution_memory]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b)
    (value : Value α d.tag) (htyped : s.mem p.input (inputOffset p s i) = .mk d.tag value) :
    (execution rest d p s).val.mem p.output (outputOffset p s i) = .mk d.tag value := by
  rw [readback rest d p s i hi hinj, htyped, Cell.read_same]

/-- Every int64 bit pattern survives selection and re-encoding, including the
signed endpoints. The source executes through its actual Int payload channel. -/
theorem int64_bits_readback {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p s i)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b)
    (bits : BitVec 64)
    (htyped : s.mem p.input (inputOffset p s i) = .mk .int bits.toInt) :
    BitVec.ofInt 64 (((execution rest .int64 p s).val.mem p.output (outputOffset p s i)).read .int) = bits := by
  rw [typed_readback rest .int64 p s i hi hinj bits.toInt htyped]
  simp [Storage.tag]

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p s i → o ≠ outputOffset p s i) :
    (execution rest d p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.masked_scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α)
    (hinj : ∀ a b, active p s a → active p s b → outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], active p s i → t.mem p.output (outputOffset p s i) =
        .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], active p s i → o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest d p s).val, (execution rest d p s).property,
    fun i hi => readback rest d p s i hi hinj, frame rest d p s⟩

#axiomsClean int64_payload_dtype
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean typed_readback
#axiomsClean int64_bits_readback
#axiomsClean frame
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
