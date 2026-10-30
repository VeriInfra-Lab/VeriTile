import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.InputBinding
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.IOSafety
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RealGrid
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.SourceGateSite

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface.Result
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

theorem real (p : Parameters) :
    (programIO Bool.false realArithmetic RealGrid.unusedGate p).ImplementsOn
      byteDomain (Reference.realFormula p) := by
  intro xs hd initial hin
  have hr := input_reads Bool.false realArithmetic RealGrid.unusedGate p xs initial hin
  have hb := hr.bytes hd
  obtain ⟨final,he,hv,hf⟩ := RealGrid.complete p initial hb
  refine ⟨fun bounds bounded => program_safe Bool.false realArithmetic RealGrid.unusedGate
    p xs bounds bounded initial hb,final,?_,?_,output_frame _ _ _ _ _ _ _ hf⟩
  · rw [program_run]; exact he
  · intro o j _
    have ho : o=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance o 0
    subst o
    change mapCell id (final.mem "OUT" (p.columns*j.1.val+j.2.1.val)) =
      Cell.mk .fp16 (LlamaFF.realTarget ⟨p.inner,p.epsilon⟩ (Reference.realInput p xs j.1.val) j.2.1.val)
    simp only [Grid.address] at hv
    rw [hv (j.1,j.2.1) trivial]
    exact congrArg (Cell.mk .fp16) (hr.real_target p j.1.val j.2.1.val)


#axiomsClean real
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface.Result
