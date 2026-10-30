import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPValues

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPMemory
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stored (M : Algebra α) (h : Call) (v : α) : Cell α :=
  .mk h.outputDType.toTileDType (ofFloat h.outputDType (M.cast (some .fp32) .real h.outputDType v))
def scatter (M : Algebra α) (h : Call) (c : Config) (row q : Nat)
    (values : Values α .real [c.BT]) (u : State α) : State α :=
  (TileShape.allIndices [c.BT]).foldl (fun t idx =>
    if Geometry.active h c q idx.1 then
      t.write h.output (h.outputStrides.address h.H row (Geometry.time h c q idx.1)) (stored M h (values idx))
    else t) u

def footprint (h : Call) (row : Nat) (cell : RegionName × Nat) : Prop :=
  cell.1=h.output ∧ ∃ t, t<h.T ∧ cell.2=h.outputStrides.address h.H row t

def Frame (h : Call) (row : Nat) (u v : State α) : Prop :=
  ∀ r off, ¬footprint h row (r,off) → u.mem r off=v.mem r off

theorem store_eval (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (row q : Nat)
    (values : Values α .real [c.BT]) (u : State α)
    (hp : u.regs .ptr [c.BT] "p_o"=some (pointer h c row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (hv : u.regs .real [c.BT] "b_o"=some values) :
    step (machine M scan) (FPProgram.storeOp h c) u=some (scatter M h c row q values u) := by
  simp only [FPProgram.storeOp,step,FP.Structural.evalExpr,FP.Structural.evalOp,hp,hm,hv,
    FP.Structural.store,FloatDType.toTileDType,Option.bind_some,Option.bind_eq_bind]
  simp only [pointer,mask,decide_eq_true_eq]
  rfl

theorem frame (M : Algebra α) (h : Call) (c : Config) (row q : Nat)
    (start : q*c.BT<h.T) (values : Values α .real [c.BT]) (u : State α) :
    Frame h row u (scatter M h c row q values u) := by
  intro r off outside
  symm
  apply State.masked_scatter_frame
  by_cases same : r=h.output
  · right
    intro i _ ha eq
    apply outside
    have ht : Geometry.time h c q i.1<h.T := by
      rw [Geometry.active_time h c q i.1 ha]
      exact (Geometry.active_iff h c q i.1 start).mp ha
    exact ⟨same,_,ht,eq⟩
  · exact Or.inl same

/-- The full tagged cell outside the output region is preserved. -/
theorem other_region (M : Algebra α) (h : Call) (c : Config) (row q : Nat)
    (values : Values α .real [c.BT]) (u : State α) (r : RegionName) (hr : r≠h.output) (off : Nat) :
    (scatter M h c row q values u).mem r off=u.mem r off := by
  exact State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hr) u

theorem cell (M : Algebra α) (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (row q : Nat)
    (hr : row<h.B*h.H) (start : q*c.BT<h.T) (values : Values α .real [c.BT]) (u : State α)
    (i : Fin c.BT) (hi : q*c.BT+i.val<h.T) :
    (scatter M h c row q values u).mem h.output (h.outputStrides.address h.H row (q*c.BT+i.val))=
      stored M h (values (i,PUnit.unit)) := by
  let idx : TileIndex [c.BT] := (i,PUnit.unit)
  have active : Geometry.active h c q i := (Geometry.active_iff h c q i start).mpr hi
  obtain ⟨before,after,he⟩ := List.append_of_mem (TileShape.mem_allIndices [c.BT] idx)
  have hn := TileShape.allIndices_nodup [c.BT]
  rw [he,List.nodup_append,List.nodup_cons] at hn
  have hafter : idx∉after := hn.2.1.1
  unfold scatter
  rw [he,List.foldl_append,List.foldl_cons,if_pos active]
  rw [State.masked_scatter_frame]
  · simp [idx,Geometry.active_time h c q i active]
  · right
    rintro ⟨k,⟨⟩⟩ hk ha eq
    have kt : Geometry.time h c q k<h.T := by
      rw [Geometry.active_time h c q k ha]
      exact (Geometry.active_iff h c q k start).mp ha
    have et := Geometry.same_row h layout row hr _ _ hi kt eq
    rw [Geometry.active_time h c q k ha] at et
    have ek : k=i := Fin.ext (by omega)
    subst k
    exact hafter hk

theorem preserves_later (M : Algebra α) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (row q : Nat) (hr : row<h.B*h.H) (start : q*c.BT<h.T)
    (values : Values α .real [c.BT]) (u : State α) (t : Nat) (ht : t<h.T) (later : (q+1)*c.BT≤t) :
    (scatter M h c row q values u).mem h.output (h.outputStrides.address h.H row t)=
      u.mem h.output (h.outputStrides.address h.H row t) := by
  apply State.masked_scatter_frame
  right
  rintro ⟨i,⟨⟩⟩ _ ha eq
  have it : Geometry.time h c q i<h.T := by
    rw [Geometry.active_time h c q i ha]
    exact (Geometry.active_iff h c q i start).mp ha
  have et := Geometry.same_row h layout row hr _ _ ht it eq
  rw [Geometry.active_time h c q i ha] at et
  have := i.isLt
  rw [Nat.add_mul,Nat.one_mul] at later
  omega

#print axioms cell
#print axioms frame
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPMemory
