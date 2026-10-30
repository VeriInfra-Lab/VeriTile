import bench.tritonbench_g.token_attn_llama2.Kernels
import bench.tritonbench_g.token_attn_llama2.Proofs.Real
import bench.tritonbench_g.token_attn_llama2.Proofs.Complete

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.RawCorrect
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Complete score-kernel grid over finite raw payload arrays, signed metadata,
source strides and an arbitrary valid launch order. All active Real outputs
and all exterior typed cells are included. -/
theorem token_attention_kernel_correct (c : Call) (m : Metadata c)
    (x : Buffers ℝ c) (rest : State ℝ) (domain : Valid c m)
    (work : List Program) (order : (programs c.grid).Perm work) :
    ∃ final, NativeGrid.runPrograms (fun _ => realAlgebra) (c.parameters (idealScale c.dim))
      c.grid work (initial c m x rest)=some final ∧
      Result c m (realFormula c (idealScale c.dim) m x) (initial c m x rest) final := by
  obtain ⟨final,run,values,frame⟩ := Whole.complete (fun _ => realAlgebra) c (idealScale c.dim)
    m x rest domain work order
  refine ⟨final,run,?_,frame⟩
  intro b hb h hh n hn
  rw [values b hb h hh n hn]
  congr 1
  exact score_initial_real c (idealScale c.dim) m x rest domain b h n hb hn

#specNonCircular realFormula avoiding [source,NativeGrid.runPrograms]
#axiomsClean token_attention_kernel_correct
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.RawCorrect
