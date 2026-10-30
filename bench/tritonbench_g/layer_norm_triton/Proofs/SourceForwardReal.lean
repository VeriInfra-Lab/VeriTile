import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardResult
import bench.tritonbench_g.layer_norm_triton.Proofs.Real

/-! Ideal-Real interpretation uses only local primitive laws. No execution,
output equation, or aggregate correctness claim is a premise. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardReal
open VeriTile.Triton FP.Structural SourceModel SourceForwardGraph
noncomputable section

structure Laws (m : Model ℝ) : Prop where
  literal : ∀ x,lit m x=x
  add : ∀ x y,bin m .add x y=x+y
  sub : ∀ x y,bin m .sub x y=x-y
  mul : ∀ x y,bin m .mul x y=x*y
  div : ∀ x y,bin m .div x y=x/y
  cast : ∀ x,cast m x=x
  sqrt : ∀ x,m.scalar.unary (some .fp32) .sqrt x=Real.sqrt x
  load : ∀ x,m.scalar.fp32Load x=x
  sum : ∀ B v,sum m B v=∑ j : Fin B,v j

theorem padded_sum (N B : Nat) (covers : N≤B) (x : Nat → ℝ) :
    (∑ j : Fin B,if j.val<N then x j.val else 0)=∑ j ∈ Finset.range N,x j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<N then x j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded x covers

theorem mean_real (m : Model ℝ) (laws : Laws m) (N B : Nat) (covers : N≤B) (x : Nat → ℝ) :
    SourceForwardGraph.mean m N B x=RealProof.Spec.mean N x := by
  simp only [SourceForwardGraph.mean,laws.div,laws.sum,xload,laws.add,laws.literal,laws.cast,zero_add]
  rw [padded_sum N B covers]
  rfl

theorem variance_real (m : Model ℝ) (laws : Laws m) (N B : Nat) (covers : N≤B) (x : Nat → ℝ) :
    SourceForwardGraph.variance m N B x=RealProof.Spec.variance N x := by
  have hc (j : Nat) : SourceForwardGraph.centered m N B x j=
      if j<N then x j-RealProof.Spec.mean N x else 0 := by
    by_cases hj : j<N <;>
      simp [SourceForwardGraph.centered,xload,mean_real m laws N B covers,laws.sub,laws.literal,laws.cast,hj]
  have hs (j : Fin B) :
      bin m .add (lit m 0) (bin m .mul (SourceForwardGraph.centered m N B x j.val) (SourceForwardGraph.centered m N B x j.val))=
        if j.val<N then (x j.val-RealProof.Spec.mean N x)^2 else 0 := by
    simp only [laws.add,laws.literal,laws.mul,hc]
    split <;> simp [pow_two]
  unfold SourceForwardGraph.variance
  simp only [laws.div,laws.sum]
  simp only [hs]
  simp only [laws.literal]
  rw [padded_sum N B covers (fun j => (x j-RealProof.Spec.mean N x)^2)]
  rfl

theorem inverse_real (m : Model ℝ) (laws : Laws m) (N B : Nat) (covers : N≤B)
    (eps : ℝ) (x : Nat → ℝ) : inverse m N B eps x=RealProof.Spec.inverse N x eps := by
  simp only [inverse,laws.div,laws.literal,laws.sqrt,laws.add,variance_real m laws N B covers]
  rfl

theorem output_real (m : Model ℝ) (laws : Laws m) (N B : Nat) (covers : N≤B)
    (eps : ℝ) (x w bias : Nat → ℝ) (j : Nat) (hj : j<N) :
    output m N B eps x w bias j=RealProof.Spec.output N x w bias eps j := by
  simp only [output,laws.cast,laws.add,laws.mul,laws.sub,laws.load,
    xload,if_pos hj,mean_real m laws N B covers,inverse_real m laws N B covers]
  rfl

theorem executes (m : Model ℝ) (laws : Laws m) (stride N B : Nat) (eps : ℝ)
    (positive : 0<N) (covers : N≤B) (s : State ℝ) :
    ∃ t,
      FP.Structural.exec (engine m)
        (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s=some t ∧
      FP.Structural.exec (engine m) (SourceCandidate.forward stride N B eps) s=some t ∧
      t.mem=(final m stride N B eps s).mem ∧
      (∀ j,j<N → t.mem "Y" (s.pids 0*stride+j)=
        .mk .real (RealProof.Spec.output N (row s stride) (vector s "W") (vector s "Bias") eps j)) ∧
      t.mem "Mean" (s.pids 0)=.mk .real (RealProof.Spec.mean N (row s stride)) ∧
      t.mem "Rstd" (s.pids 0)=.mk .real (RealProof.Spec.inverse N (row s stride) eps) ∧
      (∀ r o,SourceForwardResult.Outside stride N s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hc,hm,hy,hmu,hr,hf⟩ := SourceForwardResult.executes m stride N B eps positive covers s
  refine ⟨t,he,hc,hm,?_,?_,?_,hf⟩
  · intro j hj
    rw [hy j hj,output_real m laws N B covers eps _ _ _ j hj]
  · rw [hmu,mean_real m laws N B covers]
  · rw [hr,inverse_real m laws N B covers]

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardReal
