import bench.repairs.tritonbench_g.parallel_attention.Proofs.Wrapper
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Normalize

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

def PublicResult.Agrees (T V : Nat) : PublicResult ℝ → PublicResult ℝ → Prop
  | .both o z,.both o' z' => (∀ t,t < T → ∀ e,e < V → o t e=o' t e) ∧ (∀ t,t < T → z t=z' t)
  | .single o,.single o' => ∀ t,t < T → ∀ e,e < V → o t e=o' t e
  | _,_ => False

/-- Only active tensor elements are read or claimed; no out-of-bounds buffer
identity is needed to cover all three public return branches. -/
theorem public_read_real (s : Shape) (x : TensorInputs ℝ) (bh o z : Nat) (m : Memory ℝ)
    (eps : ℝ) (returnBoth normalize : Bool)
    (output : ∀ t,t < s.T → ∀ e,e < s.V → m (o+dense s.T s.V bh t e)=
      outputSpec s.T s.K (selectedScale s.K s.useScale) (headInputs x bh) t e)
    (normalizer : ∀ t,t < s.T → m (z+dense s.T 1 bh t 0)=
      normalizerSpec s.T s.K (selectedScale s.K s.useScale) (headInputs x bh) t) :
    PublicResult.Agrees s.T s.V (publicRead realOps eps s o z m bh returnBoth normalize)
      (publicSpec s.T s.K (selectedScale s.K s.useScale) eps (headInputs x bh) returnBoth normalize) := by
  cases returnBoth <;> cases normalize <;>
    simp only [publicRead,publicSpec,Bool.false_eq_true,if_false,if_true,PublicResult.Agrees]
  · intro t ht e he
    simpa using output t ht e he
  · intro t ht e he
    simp only [output t ht e he,normalizer t ht,real_cast,real_add]
    rfl
  · exact ⟨output,normalizer⟩
  · exact ⟨output,normalizer⟩


end
end VeriTile.Bench.Repairs.ParallelAttention
