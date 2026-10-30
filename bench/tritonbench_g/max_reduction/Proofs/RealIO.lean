import bench.tritonbench_g.max_reduction.Spec
import bench.tritonbench_g.max_reduction.Proofs.DimensionIO
import bench.tritonbench_g.max_reduction.Proofs.KernelMath
import bench.tritonbench_g.max_reduction.Proofs.Contract
import VeriTile.Triton.Float.RelationalProgramIO

namespace VeriTile.Bench.Repairs.MaxReduction.RealIO
open VeriTile.Triton FP.Structural PublicIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def observe : WithBot ℝ→ℝ | none => 0 | some value => value

theorem row_real (inp : RegionName) (M N BM BN p : Nat) (positive : 0<N) (covers : N≤BN)
    (plan : ReductionPlan BN) (xs : DimensionMath.Inputs inp BM BN) (i : Fin BM)
    (active : p*BM+i.val<M) :
    ∃ (answer : ℝ) (index : Nat),rowResult (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ))
      inp M N BM BN p xs i=((answer : WithBot ℝ),index) ∧
      RealModel.IsLeastArgmax (DimensionMath.input xs i) N answer index := by
  have same : rowResult (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ))
      inp M N BM BN p xs i=SourceGrid.target (KernelMath.engine Compute.real plan.tree)
        (fun j => (DimensionMath.input xs i j : WithBot ℝ)) N 1 BN 0 := by
    unfold rowResult SourceGrid.target
    apply congrArg ((KernelMath.engine Compute.real plan.tree).indexed BN)
    funext n
    simp [active,DimensionMath.input,n.isLt]
  rw [same]
  simpa using KernelMath.target_real (DimensionMath.input xs i) N 1 BN 0 positive covers plan

theorem row_empty (inp : RegionName) (M BM p : Nat) (plan : ReductionPlan 1)
    (xs : DimensionMath.Inputs inp BM 1) (i : Fin BM) :
    rowResult (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ))
      inp M 0 BM 1 p xs i=((⊥ : WithBot ℝ),0) := by
  have same : rowResult (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ))
      inp M 0 BM 1 p xs i=plan.tree.eval Compute.realCombine (fun _ => ((⊥ : WithBot ℝ),0)) := by
    unfold rowResult KernelMath.engine Source.treeEngine
    apply Compute.tree_congr
    intro n hn
    have bound := (plan.complete n).mp hn
    have zero : n=0 := by omega
    simp [zero,Compute.real]
  rw [same]
  have constant (t : Tree) : t.eval Compute.realCombine (fun _ => ((⊥ : WithBot ℝ),0))=((⊥ : WithBot ℝ),0) := by
    induction t with
    | leaf _ => rfl
    | node a b ia ib => simp [Tree.eval,ia,ib,Compute.realCombine]
  exact constant plan.tree

theorem satisfies (inp ov oi : RegionName) (M N K BM p k : Nat)
    (hK : 0<K) (regions : ov≠oi) (plan : ReductionPlan (nextPower N)) :
    (dimensionIO (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ)) observe
      inp ov oi M N K BM (nextPower N) p k).Satisfies
        (DimensionMath.result inp ov oi M N BM (nextPower N) p) := by
  apply ProgramIO.Satisfies.of_implements
    (dimension_implements (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ)) observe
      inp ov oi M N K BM (nextPower N) p k (nextPower_positive N) hK regions)
  intro xs i active
  by_cases empty : N=0
  · subst N
    have value := row_empty inp M BM p plan xs i
    simp only [ite_true]
    change observe (rowResult _ _ inp M 0 BM 1 p xs i).1=0 ∧
      ((rowResult _ _ inp M 0 BM 1 p xs i).2 : Int)=0
    rw [value]
    exact ⟨rfl,rfl⟩
  · obtain ⟨answer,index,value,math⟩ := row_real inp M N BM (nextPower N) p
      (by omega) (nextPower_covers N) plan xs i active
    rw [if_neg empty]
    refine ⟨index,?_,?_⟩
    · change ((rowResult _ _ inp M N BM (nextPower N) p xs i).2 : Int)=(index : Int)
      rw [value]
    · change RealModel.IsLeastArgmax (DimensionMath.input xs i) N
        (observe (rowResult _ _ inp M N BM (nextPower N) p xs i).1) index
      rw [value]
      exact math

#axiomsClean satisfies
end
end VeriTile.Bench.Repairs.MaxReduction.RealIO
