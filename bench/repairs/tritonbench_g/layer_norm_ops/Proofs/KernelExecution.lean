import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReadback
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardWhole

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelExecution
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

structure ForwardAdmissible (p : ForwardParameters) : Prop where
  weight : p.weight≠p.residualOut ∧ p.weight≠p.mean ∧ p.weight≠p.rstd
  bias : p.bias≠p.residualOut ∧ p.bias≠p.mean ∧ p.bias≠p.rstd
  res_y : p.residualOut≠p.y
  res_mu : p.residualOut≠p.mean
  res_r : p.residualOut≠p.rstd
  mu_y : p.mean≠p.y
  mu_r : p.mean≠p.rstd
  r_y : p.rstd≠p.y

structure ForwardFacts (A : Algebra α) (p : ForwardParameters) (s t : State α) : Prop where
  execution : FP.Structural.exec A (forwardKernel p) s=some t
  memory : t.mem=(Forward.reference A p s).mem
  y : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.y (s.pids 0*p.yStride+i.1.val)=Forward.stored A (KernelReference.forwardY A p s i)
  residual : p.storeResidual=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.residualOut (s.pids 0*p.residualOutStride+i.1.val)=Forward.stored A (Forward.z A p s i)
  mean : p.rms=Bool.false → t.mem p.mean (s.pids 0)=Forward.stored A (Forward.meanValue A p s)
  rstd : t.mem p.rstd (s.pids 0)=Forward.stored A (Forward.rstdValue A p s)
  frame : ∀ r o,Forward.Outside p s r o → t.mem r o=s.mem r o

structure BackwardFacts (A : Algebra α) (p : BackwardParameters) (s t : State α) : Prop where
  execution : FP.Structural.exec A (backwardKernel p) s=some t
  memory : t.mem=(Backward.reference A p s).mem
  dx : ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.dx (row*p.dxStride+i.1.val)=Forward.stored A (KernelReference.rowDX A p s row i)
  y : p.recompute=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.y (row*p.yStride+i.1.val)=Forward.stored A (KernelReference.rowY A p s row i)
  residual : p.storeDres=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.dresIn (row*p.dresInStride+i.1.val)=Forward.stored A (KernelReference.rowDX A p s row i)
  dw : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.dw (s.pids 0*p.columns+i.1.val)=Forward.stored A ((KernelReference.partials A p s).1 i)
  db : p.hasBias=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    t.mem p.db (s.pids 0*p.columns+i.1.val)=Forward.stored A ((KernelReference.partials A p s).2 i)
  frame : ∀ r o,KernelFrame.Outside p s r o → t.mem r o=s.mem r o

theorem forward (A : Algebra α) (p : ForwardParameters) (s : State α)
    (h : ForwardAdmissible p) : ∃ t,ForwardFacts A p s t := by
  obtain ⟨t,he,hm⟩ := Forward.complete_run A p s
  refine ⟨t,he,hm,?_,?_,?_,?_,?_⟩
  · intro i hi
    rw [hm,Forward.reference_y A p s i hi,KernelInvariant.forward_y A p s h.weight h.bias i hi]
  · intro hs i hi
    rw [hm]
    exact KernelReadback.forward_residual A p s hs h.res_y h.res_mu h.res_r i hi
  · intro hr
    rw [hm]
    exact Forward.reference_mean A p s hr h.mu_y h.mu_r
  · rw [hm]
    exact Forward.reference_rstd A p s h.r_y
  · intro r o hf
    rw [hm]
    exact Forward.reference_frame A p s r o hf

theorem backward (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : KernelReadback.Admissible p) : ∃ t,BackwardFacts A p s t := by
  obtain ⟨t,he,hm⟩ := Backward.complete_run A p s
  refine ⟨t,he,hm,?_,?_,?_,?_,?_,?_⟩
  · intro row lo hr i hi
    rw [hm]
    exact KernelReadback.reference_dx A p s h row lo hr i hi
  · intro hy row lo hr i hi
    rw [hm]
    exact KernelReadback.reference_y A p s h hy row lo hr i hi
  · intro hd row lo hr i hi
    rw [hm]
    exact KernelReadback.reference_dr A p s h hd row lo hr i hi
  · intro i hi
    rw [hm]
    exact KernelReadback.reference_dw A p s h i hi
  · intro hb i hi
    rw [hm]
    exact KernelReadback.reference_db A p s h hb i hi
  · intro r o hf
    rw [hm]
    exact KernelFrame.reference A p s r o hf

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelExecution
