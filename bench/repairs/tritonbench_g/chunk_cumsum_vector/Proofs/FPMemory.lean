import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPValues

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPMemory
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stored (M : Algebra α) (h : Call) (v : α) : Cell α :=
  .mk h.outputDType.toTileDType (ofFloat h.outputDType (M.cast (some .fp32) .real h.outputDType v))
def scatter (M : Algebra α) (h : Call) (c : Config) (feature row q : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α) : State α :=
  (TileShape.allIndices [c.BT,BS]).foldl (fun st idx =>
    if Geometry.active h c feature q idx.1.val idx.2.1.val then
      st.write h.output (h.outputStrides.address h.H row (Geometry.time h c q idx.1.val)
        (Geometry.column h feature idx.2.1.val)) (stored M h (values idx)) else st) u

def footprint (h : Call) (p : Geometry.Program h) (cell : RegionName × Nat) : Prop :=
  cell.1=h.output ∧ ∃ (t : Nat) (j : Fin BS), t<h.T ∧ p.feature*BS+j.val<h.S ∧
    cell.2=h.outputStrides.address h.H p.row t (p.feature*BS+j.val)
def Frame (h : Call) (p : Geometry.Program h) (u v : State α) : Prop :=
  ∀ r off, ¬footprint h p (r,off) → u.mem r off=v.mem r off

theorem store_eval (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (feature row q : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α)
    (hp : u.regs .ptr [c.BT,BS] "p_z"=some (pointer h c feature row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c feature q))
    (hv : u.regs .real [c.BT,BS] "b_c"=some values) :
    step (machine M dot) (FPProgram.storeOp h c) u=some (scatter M h c feature row q values u) := by
  simp only [FPProgram.storeOp,step,FP.Structural.evalExpr,FP.Structural.evalOp,hp,hm,hv,
    FP.Structural.store,FloatDType.toTileDType,Option.bind_some,Option.bind_eq_bind]
  simp only [pointer,mask,decide_eq_true_eq]
  rfl

theorem frame (M : Algebra α) (h : Call) (c : Config) (hc : 0<c.BT) (p : Geometry.Program h)
    (q : Nat) (hq : q<Geometry.chunks h c) (values : Values α .real [c.BT,BS]) (u : State α) :
    Frame h p u (scatter M h c p.feature p.row q values u) := by
  intro r off outside
  symm
  apply State.masked_scatter_frame
  by_cases same : r=h.output
  · right
    intro i _ ha eq
    apply outside
    have bounds := (Geometry.active_iff h c p q i.1.val i.2.1.val (Geometry.chunk_start h c hc q hq)).mp ha
    have coords := Geometry.active_coordinates h c p q i.1.val i.2.1.val ha
    exact ⟨same,q*c.BT+i.1.val,i.2.1,bounds.1,bounds.2,by simpa [coords.1,coords.2] using eq⟩
  · exact Or.inl same

theorem other_region (M : Algebra α) (h : Call) (c : Config) (feature row q : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α) (r : RegionName) (hr : r≠h.output) (off : Nat) :
    (scatter M h c feature row q values u).mem r off=u.mem r off := by
  exact State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hr) u

theorem cell (M : Algebra α) (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (hc : 0<c.BT) (p : Geometry.Program h) (q : Nat) (hq : q<Geometry.chunks h c)
    (values : Values α .real [c.BT,BS]) (u : State α) (i : Fin c.BT) (j : Fin BS)
    (ha : Geometry.active h c p.feature q i.val j.val) :
    (scatter M h c p.feature p.row q values u).mem h.output
      (h.outputStrides.address h.H p.row (q*c.BT+i.val) (p.feature*BS+j.val))=
      stored M h (values (i,j,PUnit.unit)) := by
  let idx : TileIndex [c.BT,BS] := (i,j,PUnit.unit)
  obtain ⟨before,after,he⟩ := List.append_of_mem (TileShape.mem_allIndices [c.BT,BS] idx)
  have hn := TileShape.allIndices_nodup [c.BT,BS]
  rw [he,List.nodup_append,List.nodup_cons] at hn
  have hafter : idx∉after := hn.2.1.1
  have start := Geometry.chunk_start h c hc q hq
  have bounds := (Geometry.active_iff h c p q i.val j.val start).mp ha
  have coords := Geometry.active_coordinates h c p q i.val j.val ha
  unfold scatter
  rw [he,List.foldl_append,List.foldl_cons,if_pos ha]
  rw [State.masked_scatter_frame]
  · simp [idx,coords.1,coords.2]
  · right
    rintro ⟨k,e,⟨⟩⟩ hk hm eq
    have kb := (Geometry.active_iff h c p q k.val e.val start).mp hm
    have kc := Geometry.active_coordinates h c p q k.val e.val hm
    dsimp only at eq
    rw [kc.1,kc.2] at eq
    obtain ⟨_,ht,hj⟩ := layout.unique p.row p.row _ _ _ _ p.rowInside p.rowInside bounds.1 kb.1 bounds.2 kb.2 eq
    have ek : k=i := Fin.ext (by omega)
    have ej : e=j := Fin.ext (by omega)
    subst k; subst e
    exact hafter hk

theorem preserves_earlier (M : Algebra α) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (hc : 0<c.BT) (p : Geometry.Program h) (q : Nat) (hq : q<Geometry.chunks h c)
    (values : Values α .real [c.BT,BS]) (u : State α) (t : Nat) (j : Fin BS)
    (ht : t<h.T) (hj : p.feature*BS+j.val<h.S) (earlier : t<q*c.BT) :
    (scatter M h c p.feature p.row q values u).mem h.output
      (h.outputStrides.address h.H p.row t (p.feature*BS+j.val))=
      u.mem h.output (h.outputStrides.address h.H p.row t (p.feature*BS+j.val)) := by
  apply State.masked_scatter_frame
  right
  rintro ⟨i,k,⟨⟩⟩ _ ha eq
  have bounds := (Geometry.active_iff h c p q i.val k.val (Geometry.chunk_start h c hc q hq)).mp ha
  have coords := Geometry.active_coordinates h c p q i.val k.val ha
  dsimp only at eq
  rw [coords.1,coords.2] at eq
  have same := (layout.unique p.row p.row _ _ _ _ p.rowInside p.rowInside ht bounds.1 hj bounds.2 eq).2.1
  omega

theorem scatter_undef (M : Algebra α) (h : Call) (c : Config) (feature row q : Nat)
    (values : Values α .real [c.BT,BS]) (u : State α) :
    (scatter M h c feature row q values u).undef=u.undef := by
  unfold scatter
  generalize TileShape.allIndices [c.BT,BS]=xs
  induction xs generalizing u with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    split <;> rw [ih] <;> rfl

#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPMemory
