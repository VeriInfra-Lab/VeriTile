import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Math
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Grid
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
open scoped BigOperators
namespace RealProof

theorem output_split (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk i j : Nat) (ifcond : Bool) :
    Reference.output realOps c xs scale head chunk ifcond i j =
      Body.localOutput realOps
        (Body.score realOps c (Body.query realOps c xs scale head chunk) (Body.key realOps c xs head chunk))
        (Body.value realOps c xs head chunk) i j +
      ∑ e∈Finset.range c.D, Body.query realOps c xs scale head chunk i e *
        Reference.carry realOps c xs scale head ifcond chunk e j := by
  simp only [Reference.output,Body.evaluate,Body.output]
  split_ifs with hf
  · have hn : chunk=0 := by
      by_contra hn
      simp [hn] at hf
    subst chunk
    simp [Reference.carry,realOps]
  · dsimp only
    rw [chain_real]
    rfl

theorem carry_product_padded (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk i j : Nat) (ifcond : Bool) (hr : chunk*32+i<c.T) (hj : j<c.D) :
    (∑ e∈Finset.range c.D, Body.query realOps c xs scale head chunk i e *
      Reference.carry realOps c xs scale head ifcond chunk e j)=
      scale * ∑ u∈Finset.range (chunk*32), ∑ e∈Finset.range c.D,
        xs.q head (chunk*32+i) e*MathTarget.kv c xs head u e j := by
  calc
    _ = ∑ e∈Finset.range c.D, ∑ u∈Finset.range (chunk*32),
        scale * (xs.q head (chunk*32+i) e*MathTarget.kv c xs head u e j) := by
      apply Finset.sum_congr rfl
      intro e he
      have hd := Finset.mem_range.mp he
      rw [carry_padded c xs scale head ifcond chunk e j hd hj]
      simp only [Body.query,masked,hr,hd,and_self,↓reduceIte,realOps,id_eq]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro u hu
      ring
    _ = ∑ u∈Finset.range (chunk*32), ∑ e∈Finset.range c.D,
        scale * (xs.q head (chunk*32+i) e*MathTarget.kv c xs head u e j) := Finset.sum_comm
    _ = _ := by simp only [Finset.mul_sum]

theorem output_padded (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk i j : Nat) (ifcond : Bool) (hr : chunk*32+i<c.T) (hj : j<c.D) :
    Reference.output realOps c xs scale head chunk ifcond i j =
      scale * ∑ u∈Finset.range ((chunk+1)*32), ∑ e∈Finset.range c.D,
        xs.q head (chunk*32+i) e * MathTarget.kv c xs head u e j := by
  rw [output_split,local_padded c xs scale head chunk i j hr hj,
    carry_product_padded c xs scale head chunk i j ifcond hr hj,
    Nat.succ_mul,Finset.sum_range_add,mul_add]
  ring

theorem output_target (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head row j : Nat) (ifcond : Bool) (hr : row<c.T) (hj : j<c.D) :
    Reference.output realOps c xs scale head (row/32) ifcond (row%32) j =
      MathTarget.output c xs scale head row j := by
  have hrow : row/32*32+row%32=row := by omega
  rw [output_padded c xs scale head (row/32) (row%32) j ifcond (by omega) hj,hrow]
  exact MathTarget.output_padded c xs scale head row j

theorem full_output (c : Shape) (abi : ABI c) (scale : ℝ) (store ifcond : Bool)
    (initial : Memory ℝ) (head row j : Nat) (hh : head<c.heads) (hr : row<c.T) (hj : j<c.D) :
    Source.run realOps c abi scale store ifcond initial (abi.o.base+c.mainOffset head row j)=
      MathTarget.output c (inputsOf c abi initial) scale head row j := by
  rw [Source.run_output realOps c abi scale store ifcond initial head row j hh hr hj]
  exact output_target c (inputsOf c abi initial) scale head row j ifcond hr hj

theorem full_history (c : Shape) (abi : ABI c) (scale : ℝ) (store ifcond : Bool)
    (initial : Memory ℝ) (hs : store=Bool.true) (head chunk i j : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hi : i<c.D) (hj : j<c.D) :
    Source.run realOps c abi scale store ifcond initial (abi.h.base+c.historyOffset head chunk i j)=
      MathTarget.history c (inputsOf c abi initial) head chunk i j := by
  rw [Source.run_history realOps c abi scale store ifcond initial hs head chunk i j hh hc hi hj]
  exact carry_target c (inputsOf c abi initial) scale head ifcond chunk i j hi hj

end RealProof
#axiomsClean RealProof.output_target
#axiomsClean RealProof.full_output
#axiomsClean RealProof.full_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
