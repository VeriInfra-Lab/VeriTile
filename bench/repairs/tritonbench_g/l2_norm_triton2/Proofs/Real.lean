import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Scalar
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Real
import bench.tritonbench_g.l2_norm_triton2.Proofs.Math

namespace VeriTile.Bench.Repairs.L2NormTriton2.RealExecution
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
abbrev algebra := L2NormTriton1.RealExecution.algebra

/-- Independent full logical-row gradient. Real sqrt/division are totalized;
the calculus interpretation uses a positive squared-norm-plus-epsilon. -/
def formula (N : Nat) (eps : ℝ) (xs dys : Nat → ℝ) (i : Nat) : ℝ :=
  let squared := ∑ j∈Finset.range N,(xs j)^2
  let cross := ∑ j∈Finset.range N,dys j*xs j
  dys i / Real.sqrt (squared+eps) - cross/(squared+eps)/Real.sqrt (squared+eps)*xs i

theorem dot_real (N B : Nat) (xs dys : Nat → ℝ) (cover : N≤B) :
    Tile.dot algebra N B xs dys=∑ j∈Finset.range N,dys j*xs j := by
  have filter : (Finset.range B).filter (fun i => i<N)=Finset.range N := by
    ext i
    simp only [Finset.mem_filter,Finset.mem_range]
    omega
  simp only [Tile.dot,Tile.masked,Tile.loaded,L2NormTriton1.Tile.masked,L2NormTriton1.Tile.loaded,
    algebra,L2NormTriton1.RealExecution.algebra,L2NormTriton1.RealExecution.sumValues,
    TileShape.axisDim,TileShape.insertAxisIndex,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  trans ∑ j : Fin B,if j.val<N then dys j.val*xs j.val else 0
  · apply Finset.sum_congr rfl
    intro j hj
    by_cases h : j.val<N <;> norm_num [h]
  · change (∑ j : Fin B,(fun k : Nat => if k<N then dys k*xs k else 0) j.val)=_
    rw [Fin.sum_univ_eq_sum_range (fun k : Nat => if k<N then dys k*xs k else 0)]
    rw [←Finset.sum_filter,filter]

theorem scalar_real (N B : Nat) (eps : ℝ) (xs dys : Nat → ℝ) (i : Nat) (cover : N≤B) :
    Tile.scalar algebra N B eps xs dys i=formula N eps xs dys i := by
  dsimp only [Tile.scalar,Tile.variance]
  rw [L2NormTriton1.RealExecution.variance_real N B xs cover,dot_real N B xs dys cover]
  simp only [algebra,L2NormTriton1.RealExecution.algebra,formula]
  ring

/-- Reuse the unchanged original independent mathematical helper. -/
theorem agrees_with_original_math (N : Nat) (eps : ℝ) (xs dys : Nat → ℝ) (i : Fin N) :
    formula N eps xs dys i.val=
      VeriTile.Bench.TritonBenchG.L2NormTriton2.rowBackward eps
        (fun j : Fin N => xs j.val) (fun j : Fin N => dys j.val) i := by
  simp only [formula,VeriTile.Bench.TritonBenchG.L2NormTriton2.rowBackward]
  rw [Fin.sum_univ_eq_sum_range (fun j => xs j*xs j),Fin.sum_univ_eq_sum_range (fun j => dys j*xs j)]
  simp only [pow_two]
  ring

#specNonCircular formula avoiding [Tile.scalar,Tile.dot,algebra]
#axiomsClean scalar_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.RealExecution
