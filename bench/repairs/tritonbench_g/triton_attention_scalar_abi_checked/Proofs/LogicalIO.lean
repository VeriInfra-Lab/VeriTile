import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.PublicCorrectness

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

theorem head_input (c : Shape) (xs : Inputs ℝ c) (h : Fin c.groups) :
    Erased.headTensor c (inputAt 0 c xs) "Q" h=(logicalInputs c xs).query h ∧
    Erased.headTensor c (inputAt 0 c xs) "K" h=(logicalInputs c xs).key h ∧
    Erased.headTensor c (inputAt 0 c xs) "V" h=(logicalInputs c xs).value h ∧
    Erased.headTensor c (inputAt 0 c xs) "DO" h=(logicalInputs c xs).outputCotangent h := by
  have entry (r : RegionName) (i : Fin c.rows) (d : Fin c.dim) :
      inputAt 0 c xs r (position c (h,i,d,())) =
      (if r="Q" then xs 0 (h,i,d,()) else if r="K" then xs 1 (h,i,d,())
       else if r="V" then xs 2 (h,i,d,()) else if r="DO" then xs 3 (h,i,d,())
       else if r="DQ" then xs 4 (h,i,d,()) else (0 : ℝ)) := by
    simp only [inputAt,dif_pos (position_lt c (h,i,d,())),index_position]
    rfl
  refine ⟨?_,?_,?_,?_⟩
  · funext i d
    change inputAt 0 c xs "Q" (position c (h,i,d,())) = xs 0 (h,i,d,())
    simpa using entry "Q" i d
  · funext i d
    change inputAt 0 c xs "K" (position c (h,i,d,())) = xs 1 (h,i,d,())
    simpa using entry "K" i d
  · funext i d
    change inputAt 0 c xs "V" (position c (h,i,d,())) = xs 2 (h,i,d,())
    simpa using entry "V" i d
  · funext i d
    change inputAt 0 c xs "DO" (position c (h,i,d,())) = xs 3 (h,i,d,())
    simpa using entry "DO" i d

def logicalFormula (c : Shape) (scale : ℝ) (xs : LogicalInputs c) : LogicalOutputs c :=
  ⟨fun h => Math.attention (xs.query h) (xs.key h) (xs.value h) scale,
   fun h => Math.gradientQ (xs.query h) (xs.key h) (xs.value h) (xs.outputCotangent h) scale,
   fun h => Math.gradientK (xs.query h) (xs.key h) (xs.value h) (xs.outputCotangent h) scale,
   fun h => Math.gradientV (xs.query h) (xs.key h) (xs.outputCotangent h) scale⟩

theorem lift_eq (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program) :
    (logicalRealIO c scale fw pw bw).lift (logicalFormula c scale) = realFormula c scale := by
  funext xs port i
  fin_cases port
  all_goals
    have heads := head_input c xs i.1
    dsimp [LogicalDomainIO.lift,logicalRealIO,logicalFormula,logicalOutputs,realFormula]
    rw [heads.1,heads.2.1]
  all_goals first | rw [heads.2.2.1,heads.2.2.2] | rw [heads.2.2.1] | rw [heads.2.2.2]

theorem logical_iff (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program) :
    (logicalRealIO c scale fw pw bw).ImplementsOn (logicalZeroDQ c) (logicalFormula c scale) ↔
      (realIO c scale fw pw bw).ImplementsOnCells (ZeroDQ 0 c) (realFormula c scale) := by
  have domain : (fun xs => logicalZeroDQ c ((logicalInputs c) xs)) = ZeroDQ 0 c := by
    funext xs
    apply propext
    constructor
    · intro h i; rcases i with ⟨g,r,d,u⟩; cases u; exact h g r d
    · intro h g r d; exact h (g,r,d,())
  change (realIO c scale fw pw bw).ImplementsOnCells _ _ ↔ _
  exact (congrArg₂ (realIO c scale fw pw bw).ImplementsOnCells domain
    (lift_eq c scale fw pw bw)).to_iff
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
