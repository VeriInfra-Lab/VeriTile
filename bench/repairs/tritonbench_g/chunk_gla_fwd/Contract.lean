import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Admission
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Contract

/-! The public contract is exactly the selected Python wrapper contract.
Device resource availability remains an external launch applicability boundary.
No value, tile-tail, empty-shape, alias or block-size restriction is added. -/
abbrev Request := Admission.Request
abbrev ShapeValid := Admission.ShapeValid
abbrev InputContract := Admission.Contract
abbrev check := Admission.admit
abbrev Tensor := Admission.Tensor
abbrev Mode := Admission.Mode

def Admitted (r : Request) : Prop :=
  ∃ shape : ShapeValid r.dims, InputContract r (r.dims.checked shape)

theorem admitted_iff (r : Request) :
    Admitted r ↔ ∃ a, check r = .ok a := by
  unfold Admitted check Admission.admit
  split
  · rename_i shape
    simp only [dif_pos shape]
    split
    · rename_i ok
      exact ⟨fun _ => ⟨_,rfl⟩, fun _ => ⟨shape,ok⟩⟩
    · rename_i no
      constructor
      · rintro ⟨h,ok⟩
        exact False.elim (no ok)
      · rintro ⟨a,ha⟩
        cases ha
  · rename_i no
    constructor
    · rintro ⟨h,_⟩
      exact False.elim (no h)
    · rintro ⟨a,ha⟩
      cases ha

#axiomsClean admitted_iff
end VeriTile.Bench.Repairs.ChunkGlaFwd.Contract
