import bench.repairs.tritonbench_g.token_attn_llama2.Host
import bench.repairs.tritonbench_g.token_attn_llama2.Integers
import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.WholeReal
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Mathematical full selected call: gathered QK/sqrt(D), all launched CTAs,
None return, supplied output identity, finite current storage and full frame. -/
theorem mathematical_correct (c : Call) (m : Metadata c) (x : Buffers ℝ c) (s : State ℝ)
    (abi : ABI c) (dense : Dense c) (guard : Guard c) (data : Data c m)
    (p : Placement c) (flat : Fin p.arena → Cell ℝ) (binding : FlatBinding c p m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    Formed c m ∧ ∃ result,
      invoke (fun _ => realAlgebra) c (idealScale c.dim) abi work (initial c m x s)=some result ∧
      result.returned=none ∧ result.output=abi.tensor .output ∧
      Result c m (realFormula c (idealScale c.dim) m x) (initial c m x s) result.state ∧
      FlatResult c p m (realFormula c (idealScale c.dim) m x) flat (flatten p flat result.state) := by
  have v := domain_valid c m dense guard data
  obtain ⟨result,run,ret,out,values,frame⟩ := host_result_with (fun _ => realAlgebra) c (idealScale c.dim) m x s v abi work legal
  have res : Result c m (realFormula c (idealScale c.dim) m x) (initial c m x s) result.state := by
    refine ⟨?_,frame⟩
    intro b hb h hh t ht
    rw [← score_initial_real c (idealScale c.dim) m x s v b h t hb ht]
    exact values b hb h hh t ht
  exact ⟨formed c m v,result,run,ret,out,res,finite_result c p m x s result.state flat binding v _ res⟩

#axiomsClean mathematical_correct
#auditStmt mathematical_correct
#specNonCircular realFormula avoiding [FP.Structural.exec,invoke,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
