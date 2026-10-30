import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Loop
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.NativeEpilogue

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem countedExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (current : State α) (code : List ComputeStmt)
    (hk : current.regs .int [] "K64"=some (fun _ => (p.inner : Int))) :
    step a.model (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) code) current=
      rangeInt a.model "step" 0 (steps p) 1 code current := by
  rw [step,countEvaluation a p current hk]
  simp [evalOp_unfold]

/-- Native local lowering is executed between actual shared-source stages.
Only fixed-K16 instruction components and scalar numerical instructions remain
as numerical parameters. Every program, loop iteration and store is proved. -/
theorem nativeComplete {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial : State α) (bytes : ByteValues initial) :
    ∃ final, nativeExecution a g p initial=some final ∧
      Post p initial final (nativeOutput a g p initial) := by
  let start := prefixExecution a.model p initial
  have hs := initialInvariant a Bool.true p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution a Bool.true p initial start.val 0 (by omega) hs bytes
  let finish := terminalExecution a g p initial mid hi
  refine ⟨finish.val,?_,finish.property.2⟩
  have hmid : step a.model (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)) start.val=some mid := by
    rw [countedExecution a p start.val (compiledBody p) hs.k]
    simpa [loopBody] using hm
  unfold nativeExecution
  rw [start.property]
  change (step a.model (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)) start.val).bind _=some finish.val
  rw [hmid]
  exact finish.property.1

#axiomsClean countedExecution
#axiomsClean nativeComplete
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
