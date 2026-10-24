import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.RawCorrect
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Complete shared-source execution, independent decoded Real outputs and the
entire exterior typed-memory frame, for raw kernel geometry and initial state. -/
theorem kernel_correct (p : Parameters) (initial : State ℝ) (bytes : ByteValues initial) :
    VeriTile.Spec.Real (∃ final,RealGrid.run p initial=some final ∧
      GridExecution.Post "OUT" (Grid.address p)
        (fun x : Grid.Index p => Cell.mk .fp16 (Real.target p initial x.1.val x.2.val))
        (fun _ => True) initial final) :=
  RealGrid.complete p initial bytes

#axiomsClean kernel_correct
#specNonCircular Real.target avoiding [RealGrid.run,Grid.grid,Grid.one,source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.RawCorrect
