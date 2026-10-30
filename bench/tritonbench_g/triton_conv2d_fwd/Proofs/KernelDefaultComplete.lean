import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultLoops
namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (accumulation M D p initial) current) :
    {t : State α // run (backend M D) (epilogue p) current = some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, FP.Structural.store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    ctx.memory, ctx.group, ctx.og, ctx.ba, ctx.fe, ctx.he, ctx.wi, ha, backend]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [BHW p,OF p]).foldl
    (fun t i => if active p initial i then
      t.write p.Output (outputOffset p initial i) (.mk .fp32 (outputValue M D p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (accumulation M D p initial) current) :
    (epilogueExecution M D p initial current ctx ha).val.mem = (written M D p initial).mem := by
  simp (config := {contextual := Bool.true}) [epilogueExecution, written, masked_scatter_memory, ctx.memory,
    active, outputOffset, outputValue, backend, FP.Structural.bop, Broadcast.leftIndex,
    Broadcast.rightIndex, and_assoc, and_comm, and_left_comm]

def Post {α : Type} (p : Parameters) (initial final : State α) (values : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.Output (outputOffset p initial i) = .mk .fp32 (values i)) ∧
  (∀ r o, (r ≠ p.Output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

/-- Full arbitrary-trip source execution, independent rounded reference, active
output values and the complete memory frame. No float identity, finite-value,
zero-undef, exact-dot or reassociation premise occurs. -/
theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      final.mem = (written M D p initial).mem ∧ Post p initial final (outputValue M D p initial) := by
  let start := preludeExecution M D p initial
  obtain ⟨mid, hm, hc, ha⟩ := heightLoop M D p initial start.val _
    (preludeContext M D p initial).1 (preludeContext M D p initial).2
  let finish := epilogueExecution M D p initial mid hc ha
  refine ⟨finish.val, ?_, epilogueMemory M D p initial mid hc ha, ?_, ?_⟩
  · change run (backend M D) (source p).surfaceBody initial = some finish.val
    rw [decomposition, run_append, run_append, start.property]
    simp only [Option.bind_some]
    rw [hm]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [epilogueMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault
