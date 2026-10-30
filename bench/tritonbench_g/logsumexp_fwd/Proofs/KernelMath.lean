import bench.tritonbench_g.logsumexp_fwd.Proofs.Real

namespace VeriTile.Bench.Repairs.LogsumexpFwd.KernelMath
open VeriTile.Triton TiledLogSumExp
noncomputable section
open Classical

/-- Independent tensor formula for each kernel output. Final row aggregation
is a Torch operation, not part of the Triton kernel. The formula has no maximum,
shift, execution state, floating primitive or reduction schedule. -/
def block (input : Nat → ℝ) (row segment D B : Nat) (scaled : Bool) (scale : ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => segment*B+i.val<D),
    Real.exp (if scaled then input (row*D+(segment*B+i.val))*scale
      else input (row*D+(segment*B+i.val))))

theorem block_formula (input : Nat → ℝ) (row segment D B : Nat) (positive : 0<B)
    (active : segment*B<D) (scaled : Bool) (scale : ℝ) :
    blockValue RealModel.engine (fun i => (input i : WithBot ℝ)) (scale : WithBot ℝ)
      row segment D B scaled = (block input row segment D B scaled scale : WithBot ℝ) := by
  obtain ⟨n,rfl⟩ := Nat.exists_eq_succ_of_ne_zero positive.ne'
  rw [RealModel.block_real input row segment D n scaled scale active,
    partialLSE_full_eq_blockLSE]
  apply congrArg (fun r : ℝ => (r : WithBot ℝ))
  unfold blockLSE block validLanes
  apply congrArg Real.log
  apply Finset.sum_congr rfl
  intro i hi
  have inside : segment*(n+1)+i.val<D := (Finset.mem_filter.mp hi).2
  simp [scaledLane_full,inside]

#specNonCircular block avoiding [rowBlock,Source.rowValue,blockValue]
#axiomsClean block_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.KernelMath
