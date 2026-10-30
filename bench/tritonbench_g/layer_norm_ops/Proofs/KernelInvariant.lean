import bench.tritonbench_g.layer_norm_ops.Proofs.KernelReference
import bench.tritonbench_g.layer_norm_ops.Proofs.Readback

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelInvariant
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def reads (p : BackwardParameters) : List RegionName := [p.x,p.dy,p.mean,p.rstd,p.dres]

structure Holds (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p) : Prop where
  memory : ∀ r∈reads p,a.state.mem r=s.mem r
  weight : a.weight=(Backward.initial A p s).weight
  bias : a.bias=(Backward.initial A p s).bias

theorem initial (A : Algebra α) (p : BackwardParameters) (s : State α) :
    Holds A p s (Backward.initial A p s) := ⟨fun _ _ => rfl,rfl,rfl⟩

theorem update (A : Algebra α) (p : BackwardParameters) (s : State α)
    (fresh : ∀ r∈reads p,Backward.RegionOutside p r) (row : Nat) (a : Backward.Acc α p)
    (h : Holds A p s a) : Holds A p s (Backward.update A p row a) := by
  refine ⟨?_,h.weight,h.bias⟩
  intro r hr
  funext o
  rw [Backward.update_frame A p row a r o (fresh r hr)]
  exact congrFun (h.memory r hr) o

theorem xhat (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p)
    (h : Holds A p s a) (row : Nat) :
    Backward.xhat A p a row=Backward.xhat A p (Backward.initial A p s) row := by
  funext i
  simp only [Backward.xhat,Backward.loaded,Backward.meanValue,Backward.initial,
    h.memory p.x (by simp [reads]),h.memory p.mean (by simp [reads]),h.memory p.rstd (by simp [reads])]

theorem dy (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p)
    (h : Holds A p s a) (row : Nat) :
    Backward.dy A p a row=Backward.dy A p (Backward.initial A p s) row := by
  funext i
  simp only [Backward.dy,Backward.loaded,Backward.initial,h.memory p.dy (by simp [reads])]

theorem base (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p)
    (h : Holds A p s a) (row : Nat) (i : TileIndex [p.block]) :
    Backward.dxBase A p a row i=Backward.dxBase A p (Backward.initial A p s) row i := by
  have hg : Backward.wdy A p a row=Backward.wdy A p (Backward.initial A p s) row := by
    funext j
    simp only [Backward.wdy,dy A p s a h row,h.weight]
  simp only [Backward.dxBase,Backward.c1,Backward.c2,hg,
    xhat A p s a h row,dy A p s a h row,h.weight,Backward.rstdValue,
    h.memory p.rstd (by simp [reads]),Backward.initial]

theorem dx (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p)
    (h : Holds A p s a) (row : Nat) (i : TileIndex [p.block])
    (apart : p.recompute=Bool.true → p.dres≠p.y) :
    Backward.dxValue A p a row i=KernelReference.rowDX A p s row i := by
  have hf : (Backward.recomputed A p a row).mem p.dres=a.state.mem p.dres := by
    funext o
    cases hy : p.recompute
    · simp [Backward.recomputed,hy]
    · simp only [Backward.recomputed,hy,if_true]
      exact Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl (apart hy))
  simp only [Backward.dxValue,KernelReference.rowDX,base A p s a h row i,
    Backward.loaded,hf,h.memory p.dres (by simp [reads])]

theorem y (A : Algebra α) (p : BackwardParameters) (s : State α) (a : Backward.Acc α p)
    (h : Holds A p s a) (row : Nat) (i : TileIndex [p.block]) :
    Backward.yValue A p a row i=KernelReference.rowY A p s row i := by
  simp only [Backward.yValue,KernelReference.rowY,xhat A p s a h row,h.weight,h.bias]

theorem partials (A : Algebra α) (p : BackwardParameters) (s : State α)
    (fresh : ∀ r∈reads p,Backward.RegionOutside p r) :
    ((Backward.accumulated A p s).dw,(Backward.accumulated A p s).db)=KernelReference.partials A p s := by
  apply Folds.project (Backward.update A p) (KernelReference.addRow A p s)
    (Holds A p s) (fun a => (a.dw,a.db)) (update A p s fresh) ?_ _ _ _ (initial A p s)
  intro row a h
  simp only [Backward.update,KernelReference.addRow,KernelReference.rowDW,KernelReference.rowDB,
    dy A p s a h row,xhat A p s a h row]

theorem forward_y (A : Algebra α) (p : ForwardParameters) (s : State α)
    (hw : p.weight≠p.residualOut ∧ p.weight≠p.mean ∧ p.weight≠p.rstd)
    (hb : p.bias≠p.residualOut ∧ p.bias≠p.mean ∧ p.bias≠p.rstd)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    Forward.yValue A p s i=KernelReference.forwardY A p s i := by
  simp only [Forward.yValue,KernelReference.forwardY,Forward.unspecified,
    hi,if_true,
    Forward.statistics_frame A p s p.weight i.1.val hw.1 hw.2.1 hw.2.2,
    Forward.statistics_frame A p s p.bias i.1.val hb.1 hb.2.1 hb.2.2]

#axiomsClean dx
#axiomsClean y
#axiomsClean partials
#axiomsClean forward_y
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelInvariant
