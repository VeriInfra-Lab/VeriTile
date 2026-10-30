import bench.tritonbench_g.chunk_cumsum_vector.Proofs.NativeIOLayout

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

private theorem mapValue_id (d : TileDType) (v : Value α d) : mapValue id d v = v := by
  cases d <;> rfl

theorem input_read (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (xs : (io M dot h c p).Inputs) (u : State α)
    (binding : (io M dot h c p).InputBinding xs u)
    (time : Fin h.T) (j : Fin BS) (active : p.feature*BS+j.val<h.S) :
    (u.mem h.input (h.inputStrides.address h.H p.row time.val (p.feature*BS+j.val))).read
      h.inputDType.toTileDType = xs (⟨0,by decide⟩ : Fin 1) (time,j,PUnit.unit) := by
  have bound := binding (⟨0,by decide⟩ : Fin 1) (time,j,PUnit.unit) active
  change _ = mapValue id h.inputDType.toTileDType _ at bound
  simpa only [mapValue_id] using bound

theorem loaded_eq (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (xs : (io M dot h c p).Inputs) (u : State α)
    (binding : (io M dot h c p).InputBinding xs u) (q : Nat) :
    FPValues.loaded M h c p.feature p.row u q =
      loadedInput M h c p (xs (⟨0,by decide⟩ : Fin 1)) q := by
  funext i
  unfold FPValues.loaded loadedInput
  split_ifs with active
  · have coordinates := Geometry.active_coordinates h c p q i.1.val i.2.1.val active
    have hc : p.feature*BS+i.2.1.val<h.S := by
      have bound := active.2
      unfold Geometry.activeColumn at bound
      omega
    rw [coordinates.1,coordinates.2]
    rw [input_read M dot h c p xs u binding
      ⟨q*c.BT+i.1.val,active_time h c p q _ _ active⟩ i.2.1 hc]
  · rfl

theorem carry_eq (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (xs : (io M dot h c p).Inputs) (u : State α)
    (binding : (io M dot h c p).InputBinding xs u) (q : Nat) :
    FPExecution.carry M h c p.feature p.row u q =
      carryInput M h c p (xs (⟨0,by decide⟩ : Fin 1)) q := by
  induction q with
  | zero => rfl
  | succ q ih =>
    simp only [FPExecution.carry,carryInput,loaded_eq M dot h c p xs u binding q,ih]

theorem finite_binding (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (h : Call) (c : Config) (p : Geometry.Program h) (xs : (io M dot h c p).Inputs)
    (u : State α) (binding : (io M dot h c p).InputBinding xs u)
    (domain : finiteCarry M D h c p (xs (⟨0,by decide⟩ : Fin 1))) :
    Rewrite.FiniteCarry M D h c p u := by
  intro q needed j active
  simpa only [carry_eq M dot h c p xs u binding q, loaded_eq M dot h c p xs u binding q]
    using domain q needed j active

#axiomsClean finite_binding

/-- The public logical condition is exactly the prior reached-carry condition
under the shared input binding, in both directions. -/
theorem finite_binding_iff (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (h : Call) (c : Config) (p : Geometry.Program h) (xs : (io M dot h c p).Inputs)
    (u : State α) (binding : (io M dot h c p).InputBinding xs u) :
    Rewrite.FiniteCarry M D h c p u ↔
      finiteCarry M D h c p (xs (⟨0,by decide⟩ : Fin 1)) := by
  unfold Rewrite.FiniteCarry finiteCarry
  simp only [carry_eq M dot h c p xs u binding, loaded_eq M dot h c p xs u binding]

#axiomsClean finite_binding_iff
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIO
