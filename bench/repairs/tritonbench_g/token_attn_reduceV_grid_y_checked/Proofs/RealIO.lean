import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.Interface
import bench.repairs.tritonbench_g.token_attn_reduceV.Real

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
open scoped BigOperators
noncomputable section

/-- Independent gathered weighted sum of the supplied probabilities and
value vectors. There is no softmax computation in this source. -/
def mathematical (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs ℝ p B H counts) : Outputs ℝ p B H :=
  result fun b h d => ∑ n ∈ Finset.range (counts b), probAt v b h n*valAt v b h n d

theorem weighted_logical (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs ℝ p B H counts) (s : State ℝ) (b : Fin B) (h : Fin H)
    (binding : CallInputs p B H counts v s b h) (positive : 0<p.block) (d : Fin p.width) :
    Real.weighted p s d.val = ∑ n ∈ Finset.range (counts b), probAt v b h n*valAt v b h n d := by
  unfold Real.weighted
  rw [binding.length]
  apply Finset.sum_congr rfl
  intro n hn
  rw [binding.probability n (Finset.mem_range.mp hn) (Nat.ne_of_gt positive),
    binding.value n (Finset.mem_range.mp hn) (Nat.ne_of_gt positive) d]

theorem real_implements (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (jobs : List Grid.PID) (order : (Grid.fullGrid B H).Perm jobs)
    (positive : 0<p.block) (separate : Grid.Separate p) (unique : Grid.Unique p jobs) :
    (programIO Real.algebra Real.sum p B H counts jobs).ImplementsOn
      (LengthsMatch p B H counts) (mathematical p B H counts) := by
  intro v domain bounds bounded initial inputs
  obtain ⟨safe,final,ran,values,frame⟩ := implements Real.algebra Real.sum p B H counts jobs order separate unique
    v domain bounds bounded initial inputs
  refine ⟨safe,final,ran,?_,frame⟩
  intro q j active
  have hq : q.val<1 := q.isLt
  have eq : q=⟨0,by simp [programIO,memory,outputs]⟩ := Fin.ext (show q.val=0 by omega)
  subst q
  have binding := call_inputs Real.algebra Real.sum p B H counts jobs v domain
    (Grid.state initial.mem (j.1.val,j.2.1.val)) inputs j.1 j.2.1 rfl rfl
  have formula := (source_values Real.algebra Real.sum p B H counts v _ j.1 j.2.1 binding j.2.2.1).symm.trans
    ((Real.output_weighted p _ positive (j.2.2.1,PUnit.unit)).trans
      (weighted_logical p B H counts v _ j.1 j.2.1 binding positive j.2.2.1))
  exact (values ⟨0,by simp [programIO,memory,outputs]⟩ j active).trans (congrArg (Cell.mk .fp32) formula)

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
