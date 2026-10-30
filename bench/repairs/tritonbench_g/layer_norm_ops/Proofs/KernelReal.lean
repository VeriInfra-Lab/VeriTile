import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalLoop

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 2000000

def readRow (s : State ℝ) (r : RegionName) (stride row col : Nat) : ℝ :=
  (s.mem r (row*stride+col)).read .fp32
def rowDX (p : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  Logical.dx p.columns p.rms p.hasDres (readRow s p.x p.xStride row)
    (fun j => (s.mem p.weight j).read .fp32) (readRow s p.dy p.dyStride row)
    (readRow s p.dres p.dresStride row) ((s.mem p.mean (row*p.meanStride)).read .fp32)
    ((s.mem p.rstd (row*p.rstdStride)).read .fp32) col
def rowY (p : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  (readRow s p.x p.xStride row col-if p.rms then 0 else (s.mem p.mean (row*p.meanStride)).read .fp32)*
    (s.mem p.rstd (row*p.rstdStride)).read .fp32*(s.mem p.weight col).read .fp32+
    if p.hasBias then (s.mem p.bias col).read .fp32 else 0
def partialDW (p : BackwardParameters) (s : State ℝ) (col : Nat) : ℝ :=
  ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s),Logical.rowDW p s row col
def partialDB (p : BackwardParameters) (s : State ℝ) (col : Nat) : ℝ :=
  ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s),Logical.rowDB p s row col

theorem forward_value (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns≤p.block) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelReference.forwardY (Logical.algebra rest) p s i=
      Logical.y p.columns p.rms p.hasBias p.epsilon (Logical.input p s)
        (fun j => (s.mem p.weight j).read .fp32) (fun j => (s.mem p.bias j).read .fp32) i.1.val := by
  unfold KernelReference.forwardY Forward.unspecified
  rw [Logical.z_value,Logical.mean_value rest p s hb,Logical.rstd_value rest p s hb]
  cases hr : p.rms <;> cases hbb : p.hasBias <;>
    simp [hi,hr,hbb,Logical.y,Logical.normalized,Logical.mean,Forward.cast,Forward.unhintedBinary,
      Logical.algebra,LayerNormLiger.Logical.algebra]

theorem dx_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (hb : p.columns≤p.block) (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelReference.rowDX (Logical.algebra rest) p s row i=rowDX p s row i.1.val := by
  have hw : ∀ j : TileIndex [p.block],j.1.val<p.columns →
      (Backward.initial (Logical.algebra rest) p s).weight j=(s.mem p.weight j.1.val).read .fp32 := by
    intro j hj
    simp [Backward.initial,Backward.loaded,Forward.cast,Logical.algebra,LayerNormLiger.Logical.algebra,hj]
  unfold KernelReference.rowDX
  rw [Logical.backward_dxBase rest p _ row (fun j => (s.mem p.weight j).read .fp32) hw hb i hi]
  cases hd : p.hasDres <;>
    simp [rowDX,readRow,hd,Backward.loaded,hi,Forward.cast,Forward.binary,Backward.initial,
      Logical.algebra,LayerNormLiger.Logical.algebra,Logical.dx,Logical.bx,Logical.bdy,Logical.bmu,Logical.binv]

theorem y_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelReference.rowY (Logical.algebra rest) p s row i=rowY p s row i.1.val := by
  unfold KernelReference.rowY Backward.yValue
  rw [Logical.backward_xhat]
  cases hb : p.hasBias <;>
    simp [rowY,readRow,Logical.bhat,Logical.bx,Logical.bmu,Logical.binv,
      Backward.initial,Backward.loaded,Forward.cast,Forward.unhintedBinary,
      Logical.algebra,LayerNormLiger.Logical.algebra,hi,hb]

theorem inputs_fresh (p : BackwardParameters) (h : KernelReadback.Admissible p) : Logical.InputsFresh p := by
  intro r hr
  apply h.fresh r
  simpa [KernelInvariant.reads] using List.mem_append_left [p.dres] hr

theorem dw_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (h : KernelReadback.Admissible p) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (KernelReference.partials (Logical.algebra rest) p s).1 i=partialDW p s i.1.val := by
  rw [←KernelInvariant.partials (Logical.algebra rest) p s h.fresh]
  exact Logical.loop_dw rest p s (inputs_fresh p h) i hi

theorem db_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (h : KernelReadback.Admissible p) (hb : p.hasBias=Bool.true)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (KernelReference.partials (Logical.algebra rest) p s).2 i=partialDB p s i.1.val := by
  rw [←KernelInvariant.partials (Logical.algebra rest) p s h.fresh]
  exact Logical.loop_db rest p s (inputs_fresh p h) i hi hb

theorem stored_read (rest : Algebra ℝ) (x : ℝ) :
    (Forward.stored (Logical.algebra rest) x).read .fp32=x := rfl

structure ForwardOutputs (p : ForwardParameters) (s t : State ℝ) : Prop where
  y : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.y (s.pids 0*p.yStride+i.1.val)).read .fp32=
      Logical.y p.columns p.rms p.hasBias p.epsilon (Logical.input p s)
        (fun j => (s.mem p.weight j).read .fp32) (fun j => (s.mem p.bias j).read .fp32) i.1.val
  residual : p.storeResidual=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.residualOut (s.pids 0*p.residualOutStride+i.1.val)).read .fp32=Logical.input p s i.1.val
  mean : p.rms=Bool.false → (t.mem p.mean (s.pids 0)).read .fp32=
    Logical.mean p.columns p.rms (Logical.input p s)
  rstd : (t.mem p.rstd (s.pids 0)).read .fp32=
    Logical.rstd p.columns p.rms p.epsilon (Logical.input p s)

structure BackwardOutputs (p : BackwardParameters) (s t : State ℝ) : Prop where
  dx : ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.dx (row*p.dxStride+i.1.val)).read .fp32=rowDX p s row i.1.val
  y : p.recompute=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.y (row*p.yStride+i.1.val)).read .fp32=rowY p s row i.1.val
  residual : p.storeDres=Bool.true → ∀ row,Backward.start p s≤row → row<Backward.stop p s →
    ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.dresIn (row*p.dresInStride+i.1.val)).read .fp32=rowDX p s row i.1.val
  dw : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.dw (s.pids 0*p.columns+i.1.val)).read .fp32=partialDW p s i.1.val
  db : p.hasBias=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (t.mem p.db (s.pids 0*p.columns+i.1.val)).read .fp32=partialDB p s i.1.val

theorem forward (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (h : KernelExecution.ForwardAdmissible p) (hb : p.columns≤p.block) :
    ∃ t,KernelExecution.ForwardFacts (Logical.algebra rest) p s t ∧ ForwardOutputs p s t := by
  obtain ⟨t,ht⟩ := KernelExecution.forward (Logical.algebra rest) p s h
  refine ⟨t,ht,?_,?_,?_,?_⟩
  · intro i hi
    rw [ht.y i hi,stored_read,forward_value rest p s hb i hi]
  · intro hs i hi
    rw [ht.residual hs i hi,stored_read,Logical.z_value,if_pos hi]
  · intro hr
    rw [ht.mean hr,stored_read,Logical.mean_value rest p s hb]
    simp [Logical.mean,hr]
  · rw [ht.rstd,stored_read,Logical.rstd_value rest p s hb]

theorem backward (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (h : KernelReadback.Admissible p) (hb : p.columns≤p.block) :
    ∃ t,KernelExecution.BackwardFacts (Logical.algebra rest) p s t ∧ BackwardOutputs p s t := by
  obtain ⟨t,ht⟩ := KernelExecution.backward (Logical.algebra rest) p s h
  refine ⟨t,ht,?_,?_,?_,?_,?_⟩
  · intro row lo hr i hi
    rw [ht.dx row lo hr i hi,stored_read,dx_value rest p s hb row i hi]
  · intro hy row lo hr i hi
    rw [ht.y hy row lo hr i hi,stored_read,y_value rest p s row i hi]
  · intro hd row lo hr i hi
    rw [ht.residual hd row lo hr i hi,stored_read,dx_value rest p s hb row i hi]
  · intro i hi
    rw [ht.dw i hi,stored_read,dw_value rest p s h i hi]
  · intro hy i hi
    rw [ht.db hy i hi,stored_read,db_value rest p s h hy i hi]

#axiomsClean forward
#axiomsClean backward
#specNonCircular rowDX avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular rowY avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular partialDW avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#specNonCircular partialDB avoiding [forwardKernel,backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelReal
