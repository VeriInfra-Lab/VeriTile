import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelMemory
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def footprint (kind : Kind) (c : Shape) (p : Program c) (r : RegionName) (off : Nat) : Prop :=
  match kind with
  | .forward => r="O" ∧ ∃ t : Fin c.N,∃ e : Fin 64,∃ j : Fin 16,off=address c p t.val e.val j.val
  | .backward =>
      (r="DI" ∧ ∃ t : Fin (c.N-1),∃ e : Fin 64,∃ j : Fin 16,off=address c p t.val e.val j.val) ∨
      (r="DG" ∧ ∃ t : Fin (c.N-1),off=gradientAddress c p t.val) ∨
      (r="DL" ∧ ∃ e : Fin 64,∃ j : Fin 16,off=dseedAddress c p e.val j.val)
def Frame (kind : Kind) (c : Shape) (p : Program c) (u v : State α) : Prop :=
  ∀ r off,¬footprint kind c p r off → v.mem r off=u.mem r off

theorem scatter_cell (c : Shape) (p : Program c) (r : RegionName) (dtype : FloatDType)
    (t : Nat) (value : Lane → α) (u : State α) (e : Fin 64) (j : Fin 16) :
    (scatter r dtype (fun a b => address c p t a.val b.val) value u).mem r (address c p t e.val j.val)=
      .mk dtype.toTileDType (ofFloat dtype (value (e,j,PUnit.unit))) := by
  apply State.scatter_readback (shape:=[64,16]) r
    (fun i => address c p t i.1.val i.2.1.val)
    (fun i => Cell.mk dtype.toTileDType (ofFloat dtype (value i))) u (e,j,PUnit.unit)
  rintro ⟨a,b,⟨⟩⟩ ⟨a',b',⟨⟩⟩ same
  obtain ⟨_,ha,hb⟩ := address_unique c p t t a a' b b' same
  subst a';subst b';rfl

theorem scatter_other_time (c : Shape) (p : Program c) (r : RegionName) (dtype : FloatDType)
    (t v : Nat) (different : t≠v) (value : Lane → α) (u : State α) (e : Fin 64) (j : Fin 16) :
    (scatter r dtype (fun a b => address c p t a.val b.val) value u).mem r (address c p v e.val j.val)=
      u.mem r (address c p v e.val j.val) := by
  apply State.scatter_frame
  apply Or.inr
  intro i _ same
  exact different (address_unique c p t v i.1 e i.2.1 j same.symm).1

theorem forward_frame (c : Shape) (p : Program c) (t : Fin c.N)
    (value : Lane → α) (u : State α) :
    Frame .forward c p u (scatter "O" .real (fun e j => address c p t.val e.val j.val) value u) := by
  intro r off outside
  apply State.scatter_frame
  by_cases same : r="O"
  · subst r
    apply Or.inr
    intro i _ equal
    exact outside ⟨rfl,t,i.1,i.2.1,equal⟩
  · exact Or.inl same

theorem gradient_unique (c : Shape) (p : Program c) (t v : Nat)
    (same : gradientAddress c p t=gradientAddress c p v) : t=v := by
  have positive := c.tiles_pos
  have products : t*c.tiles=v*c.tiles := by
    unfold gradientAddress at same
    unfold Shape.tiles
    nlinarith [same]
  exact Nat.eq_of_mul_eq_mul_right positive products

#axiomsClean scatter_cell
#axiomsClean forward_frame
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelMemory
