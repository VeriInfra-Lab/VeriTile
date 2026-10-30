import bench.tritonbench_g.kcache_copy_triton.Proofs.SignedCopy

/-! The same arbitrary-token source template with discrete data storage.
Data is only loaded and stored: signed/unsigned widths select the source's
storage declaration, and the result preserves every well-typed initial value.
Metadata/address arithmetic retains the signed source model and its separately
documented native integer-width boundary. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedDiscreteCopy
open Lean VeriTile.Triton FP.Structural
open SignedCopy (Parameters inputOffset outputOffset)
open TypedCopy (copyBackend scatter_memory)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | boolean
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .i8 | .i16 | .i32 | .i64 => .int
  | .u8 | .u16 | .u32 | .u64 => .nat
  | .boolean => .bool

def source : Storage → Parameters → ComputeKernel
  | .i8 => kcacheSignedSource%[tl.int8]
  | .i16 => kcacheSignedSource%[tl.int16]
  | .i32 => kcacheSignedSource%[tl.int32]
  | .i64 => kcacheSignedSource%[tl.int64]
  | .u8 => kcacheSignedSource%[tl.uint8]
  | .u16 => kcacheSignedSource%[tl.uint16]
  | .u32 => kcacheSignedSource%[tl.uint32]
  | .u64 => kcacheSignedSource%[tl.uint64]
  | .boolean => kcacheSignedSource%[tl.int1]

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl
    (fun t i => t.write p.KCache (outputOffset p s i) (loaded d p s i)) s

/-- Shape sanity check for the shared source template above. Equal statement
counts alone do not establish source correspondence. -/
theorem source_statement_count (d : Storage) (p : Parameters) :
    (source d p).surfaceBody.length = (SignedCopy.source .fp16 p).surfaceBody.length := by
  cases d <;> rfl

def execution {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (source d p) s = some t} := by
  cases d <;> (refine ⟨?_, ?_⟩
               swap
               simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.numericLt, copyBackend, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
               try rfl)

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    (execution rest d p s).val.mem = (copied d p s).mem := by
  cases d <;> simp [execution, copied, scatter_memory, loaded, Storage.tag,
    inputOffset, outputOffset, SignedCopy.blockId, SignedCopy.past,
    SignedCopy.sequence, SignedCopy.tokenShift, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv, integralMod,
    SignedCopy.unsigned64, ← IntegerExpr.tdiv_formula, Int.tmod_def, Region.cast]

theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) :
    (execution rest d p s).val.mem p.KCache (outputOffset p s i) = loaded d p s i := by
  rw [execution_memory]
  exact State.scatter_readback _ _ _ _ i (SignedCopy.output_injective p s)

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) :
    (execution rest d p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  apply State.scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

theorem typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (value : Value α d.tag)
    (htyped : s.mem p.K (inputOffset p s i) = .mk d.tag value) :
    (execution rest d p s).val.mem p.KCache (outputOffset p s i) = .mk d.tag value := by
  rw [readback]
  simp only [loaded, htyped, Cell.read_same]

/-- Both whole-memory value and frame contracts use only the initial state;
no numerical primitive, execution result or correctness premise occurs there. -/
theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      t.mem = (copied d p s).mem ∧
      (∀ i : TileIndex [p.width], t.mem p.KCache (outputOffset p s i) = loaded d p s i) ∧
      (∀ r o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest d p s).val, (execution rest d p s).property,
    execution_memory rest d p s, readback rest d p s, frame rest d p s⟩

#specNonCircular copied avoiding [source, execution, FP.Structural.exec]
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedDiscreteCopy
