import bench.tritonbench_g.fast_layernorm.Proofs.Real

namespace VeriTile.Bench.Repairs.FastLayernorm.RealProof
noncomputable section

theorem mean_congr (N : Nat) (x y : Nat → ℝ) (equal : ∀ j,j<N → x j=y j) :
    meanSpec N x=meanSpec N y := by
  unfold meanSpec
  congr 1
  exact Finset.sum_congr rfl fun j hj => equal j (Finset.mem_range.mp hj)

theorem variance_congr (N : Nat) (x y : Nat → ℝ) (equal : ∀ j,j<N → x j=y j) :
    varianceSpec N x=varianceSpec N y := by
  unfold varianceSpec
  rw [mean_congr N x y equal]
  congr 1
  exact Finset.sum_congr rfl fun j hj => by rw [equal j (Finset.mem_range.mp hj)]

theorem inv_congr (N : Nat) (x y : Nat → ℝ) (equal : ∀ j,j<N → x j=y j) (eps : ℝ) :
    invSpec N x eps=invSpec N y eps := by
  unfold invSpec
  rw [variance_congr N x y equal]

theorem output_congr (N : Nat) (x x' w w' b b' : Nat → ℝ) (eps : ℝ) (col : Nat)
    (hc : col<N) (hx : ∀ j,j<N → x j=x' j) (hw : w col=w' col) (hb : b col=b' col) :
    outputSpec N x w b eps col=outputSpec N x' w' b' eps col := by
  unfold outputSpec
  rw [hx col hc,mean_congr N x x' hx,inv_congr N x x' hx,hw,hb]

theorem gradient_congr (N : Nat) (x x' dy dy' w w' : Nat → ℝ) (avg inv : ℝ) (col : Nat)
    (hc : col<N) (hx : ∀ j,j<N → x j=x' j) (hdy : ∀ j,j<N → dy j=dy' j)
    (hw : ∀ j,j<N → w j=w' j) :
    gradientSpec N x dy w avg inv col=gradientSpec N x' dy' w' avg inv col := by
  have q : ∀ j,j<N → dy j*w j=dy' j*w' j := fun j hj => by rw [hdy j hj,hw j hj]
  have first : (∑ j ∈ Finset.range N,dy j*w j)=∑ j ∈ Finset.range N,dy' j*w' j :=
    Finset.sum_congr rfl fun j hj => q j (Finset.mem_range.mp hj)
  have second : (∑ j ∈ Finset.range N,dy j*w j*((x j-avg)*inv))=
      ∑ j ∈ Finset.range N,dy' j*w' j*((x' j-avg)*inv) := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [q j (Finset.mem_range.mp hj),hx j (Finset.mem_range.mp hj)]
  simp only [gradientSpec]
  rw [q col hc,hx col hc,first,second]

#axiomsClean output_congr
#axiomsClean gradient_congr
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.RealProof
