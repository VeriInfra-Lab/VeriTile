import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Window
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Range
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Epilogue

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
variable {α : Type} [Inhabited α]

/-- Successful complete candidate execution is a conclusion. Its final
memory is the independent six-input recurrence followed by the exact padded
L/M/O stores. No execution, output or whole-dot equation is a premise. -/
theorem exec_complete (c : Call) (o : Ops α) (positive : 0<c.BN) (s : State α) :
    ∃ t,exec (engine o.algebra) (Kernels.source c) s=some t ∧
      t.mem=(scatter c o (inputs s) (result c o (inputs s)) s).mem := by
  obtain ⟨pre,hpre,live,memory⟩ := prelude c o s
  obtain ⟨post,hloop,live',memory'⟩ := loop c o positive s pre live memory
  obtain ⟨final,hfinal,effect⟩ := epilogue c o (inputs s) (result c o (inputs s)) post live'
  refine ⟨final,?_,?_⟩
  · change run _ (Kernels.source c).surfaceBody s=some final
    rw [source_split,run_append,run_append,hpre]
    simp only [Option.bind_some,run,hloop]
    exact hfinal
  · rw [effect]
    exact scatter_mem_congr c o (inputs s) (result c o (inputs s)) post s memory'

#axiomsClean exec_complete
#specNonCircular result avoiding [exec_complete,FP.Structural.exec,FP.Structural.run,Kernels.source,parts]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
