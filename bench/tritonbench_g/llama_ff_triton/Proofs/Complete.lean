import bench.tritonbench_g.llama_ff_triton.Proofs.Loop
import bench.tritonbench_g.llama_ff_triton.Proofs.Epilogue
import bench.tritonbench_g.llama_ff_triton.Proofs.NativeComplete

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

/-- Full shared-quotation execution, including every K tail and the actual
masked half store. The result and frame use initial-memory arithmetic and
contain no assumed source/program equality or whole-result premise. -/
theorem sourceComplete {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial : State α) (bytes : ByteValues initial) :
    ∃ final, FP.Structural.exec a.model (source p) initial=some final ∧
      Post p initial final (sourceOutput a p initial) := by
  let start := prefixExecution a.model p initial
  have hs := initialInvariant a Bool.false p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution a Bool.false p initial start.val 0 (by omega) hs bytes
  let finish := epilogueExecution a p initial mid hi
  refine ⟨finish.val,?_,epiloguePost a p initial mid hi⟩
  change run a.model (source p).surfaceBody initial=some finish.val
  rw [partition,run_append,run_append,start.property]
  simp only [Option.bind_some]
  have hloop : run a.model [.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (body p)] start.val=some mid := by
    rw [run,countedExecution a p start.val (body p) hs.k]
    have hm' : rangeInt a.model "step" 0 (steps p) 1 (body p) start.val=some mid := by
      simpa [loopBody] using hm
    rw [hm']
    simp [run]
  rw [hloop]
  exact finish.property

#axiomsClean sourceComplete
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
