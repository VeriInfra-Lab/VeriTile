import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPValues

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPMemory
open VeriTile.Triton FP.Structural FPValues FPIndices
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stored (M : Algebra α) (h : Call) (v : α) : Cell α :=
  .mk h.outputDType.toTileDType (ofFloat h.outputDType (M.cast (some .fp32) .real h.outputDType v))

def scatter (M : Algebra α) (h : Call) (c : Config) (row q f : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α) : State α :=
  (TileShape.allIndices [c.BT,BS]).foldl (fun t idx =>
    if Geometry.active h c q f idx.1 idx.2.1 then
      t.write h.output (Geometry.address h row (Geometry.time h c q idx.1) (Geometry.feature h f idx.2.1)) (stored M h (values idx))
    else t) u

def footprint (h : Call) (row f : Nat) (cell : RegionName × Nat) : Prop :=
  cell.1=h.output ∧ ∃ t, ∃ j : Fin BS, t<h.T ∧ f*BS+j.val<h.S ∧ cell.2=Geometry.address h row t (f*BS+j.val)

def Frame (h : Call) (row f : Nat) (u v : State α) : Prop :=
  ∀ r off, ¬footprint h row f (r,off) → u.mem r off=v.mem r off

theorem store_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (row q f : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α)
    (hp : u.regs .ptr [c.BT,BS] "p_z"=some (pointer h c row q f h.output))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f))
    (hv : u.regs .real [c.BT,BS] "b_c"=some values) :
    step (machine M dot) (FPProgram.storeOp h c) u=some (scatter M h c row q f values u) := by
  simp only [FPProgram.storeOp,step,FP.Structural.evalExpr,FP.Structural.evalOp,hp,hm,hv,
    FP.Structural.store,FloatDType.toTileDType,Option.bind_some,Option.bind_eq_bind]
  simp only [pointer,mask,decide_eq_true_eq]
  rfl

theorem frame (M : Algebra α) (h : Call) (c : Config) (row q : Nat) (f : Geometry.FeatureBlock h)
    (start : q*c.BT<h.T) (values : Values α .real [c.BT,BS]) (u : State α) :
    Frame h row f.val u (scatter M h c row q f.val values u) := by
  intro r off outside
  symm
  apply State.masked_scatter_frame
  by_cases same : r=h.output
  · right
    intro i _ ha eq
    have hv := (Geometry.active_iff h c q f i.1 i.2.1 start).mp ha
    apply outside
    refine ⟨same,q*c.BT+i.1.val,i.2.1,hv.1,hv.2,?_⟩
    simpa [Geometry.time,Geometry.feature,ha.1,ha.2] using eq
  · exact Or.inl same

theorem other_region (M : Algebra α) (h : Call) (c : Config) (row q f : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α) (r : RegionName) (hr : r≠h.output) (off : Nat) :
    (scatter M h c row q f values u).mem r off=u.mem r off :=
  State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hr) u

theorem cell (M : Algebra α) (h : Call) (c : Config) (row : Geometry.Row h) (q : Nat)
    (f : Geometry.FeatureBlock h) (start : q*c.BT<h.T) (values : Values α .real [c.BT,BS]) (u : State α)
    (i : Fin c.BT) (j : Fin BS) (hi : q*c.BT+i.val<h.T) (hj : f.val*BS+j.val<h.S) :
    (scatter M h c row.val q f.val values u).mem h.output (Geometry.address h row.val (q*c.BT+i.val) (f.val*BS+j.val))=
      stored M h (values (i,j,PUnit.unit)) := by
  let idx : TileIndex [c.BT,BS] := (i,j,PUnit.unit)
  have active : Geometry.active h c q f.val i j := (Geometry.active_iff h c q f i j start).mpr ⟨hi,hj⟩
  obtain ⟨before,after,he⟩ := List.append_of_mem (TileShape.mem_allIndices [c.BT,BS] idx)
  have hn := TileShape.allIndices_nodup [c.BT,BS]
  rw [he,List.nodup_append,List.nodup_cons] at hn
  have hafter : idx∉after := hn.2.1.1
  unfold scatter
  rw [he,List.foldl_append,List.foldl_cons,if_pos active]
  rw [State.masked_scatter_frame]
  · simp [idx,Geometry.time,Geometry.feature,active.1,active.2]
  · right
    rintro ⟨k,l,⟨⟩⟩ hk ha eq
    have hv := (Geometry.active_iff h c q f k l start).mp ha
    simp only [Geometry.time,Geometry.feature,if_pos ha.1,if_pos ha.2] at eq
    have unique := Geometry.unique h row row _ _ _ _ hi hv.1 hj hv.2 eq
    have ek : k=i := Fin.ext (by omega)
    have el : l=j := Fin.ext (by omega)
    subst k
    subst l
    exact hafter hk

theorem preserves_later (M : Algebra α) (h : Call) (c : Config) (row : Geometry.Row h) (q : Nat)
    (f : Geometry.FeatureBlock h) (start : q*c.BT<h.T) (values : Values α .real [c.BT,BS]) (u : State α)
    (t s : Nat) (ht : t<h.T) (hs : s<h.S) (later : (q+1)*c.BT≤t) :
    (scatter M h c row.val q f.val values u).mem h.output (Geometry.address h row.val t s)=
      u.mem h.output (Geometry.address h row.val t s) := by
  apply State.masked_scatter_frame
  right
  rintro ⟨i,j,⟨⟩⟩ _ ha eq
  have hv := (Geometry.active_iff h c q f i j start).mp ha
  simp only [Geometry.time,Geometry.feature,if_pos ha.1,if_pos ha.2] at eq
  have et := (Geometry.unique h row row _ _ _ _ ht hv.1 hs hv.2 eq).2.1
  have := i.isLt
  rw [Nat.add_mul,Nat.one_mul] at later
  omega

#print axioms cell
#print axioms frame
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPMemory
