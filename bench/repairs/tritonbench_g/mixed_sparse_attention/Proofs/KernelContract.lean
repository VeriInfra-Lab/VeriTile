import bench.repairs.tritonbench_g.mixed_sparse_attention.HostContract

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelContract
open VeriTile Triton Whole002 Source Whole Checked

/-- Finite signed metadata checks, independent of source execution and output.
Payload readers may inspect actual int32 storage without copying float inputs. -/
def SparseChecks (c : Config) (v : Observed) : Prop :=
  (∀ b : Fin c.Z,0≤v .sequence b.val ∧ v .sequence b.val≤c.T) ∧
  (∀ h : Fin (c.Z*c.H),∀ r : Fin c.NR,
    0≤v .blocks (h.val*c.NR+r.val) ∧ v .blocks (h.val*c.NR+r.val)≤c.NS) ∧
  (∀ h : Fin (c.Z*c.H),∀ r : Fin c.NR,
    0≤v .counts (h.val*c.NR+r.val) ∧ v .counts (h.val*c.NR+r.val)≤c.NV) ∧
  (∀ h : Fin (c.Z*c.H),∀ r : Fin c.NR,∀ j : Fin c.NS,
    j.val<(v .blocks (h.val*c.NR+r.val)).toNat →
    0≤v .offsets ((h.val*c.NR+r.val)*c.NS+j.val) ∧
    v .offsets ((h.val*c.NR+r.val)*c.NS+j.val)<v .sequence (h.val/c.H)) ∧
  (∀ h : Fin (c.Z*c.H),∀ r : Fin c.NR,∀ j : Fin c.NV,
    j.val<(v .counts (h.val*c.NR+r.val)).toNat →
    0≤v .columns ((h.val*c.NR+r.val)*c.NV+j.val) ∧
    v .columns ((h.val*c.NR+r.val)*c.NV+j.val)<v .sequence (h.val/c.H))
instance (c : Config) (v : Observed) : Decidable (SparseChecks c v) := by
  unfold SparseChecks
  infer_instance
def checkSparse (c : Config) (v : Observed) : Bool := decide (SparseChecks c v)

theorem sparse_iff (c : Config) (v : Observed) :
    checkSparse c v=Bool.true ↔ MetadataGuard c v := by
  rw [checkSparse,decide_eq_true_eq]
  constructor
  · rintro ⟨hs,hb,hc,ho,hi⟩
    refine ⟨fun b h=>hs ⟨b,h⟩,fun h hh r hr=>hb ⟨h,hh⟩ ⟨r,hr⟩,
      fun h hh r hr=>hc ⟨h,hh⟩ ⟨r,hr⟩,?_,?_⟩
    · intro h hh r hr j hj
      have count:=hb ⟨h,hh⟩ ⟨r,hr⟩
      exact ho ⟨h,hh⟩ ⟨r,hr⟩ ⟨j,by dsimp only at count;omega⟩ hj
    · intro h hh r hr j hj
      have count:=hc ⟨h,hh⟩ ⟨r,hr⟩
      exact hi ⟨h,hh⟩ ⟨r,hr⟩ ⟨j,by dsimp only at count;omega⟩ hj
  · intro h
    exact ⟨fun b=>h.sequence b.val b.isLt,
      fun b r=>h.blocks b.val b.isLt r.val r.isLt,
      fun b r=>h.counts b.val b.isLt r.val r.isLt,
      fun b r j=>h.offsets b.val b.isLt r.val r.isLt j.val,
      fun b r j=>h.columns b.val b.isLt r.val r.isLt j.val⟩

/-- Existing checked metadata and storage schema, used only as an entry
contract. There is no fresh-allocation event or host-success premise. Output
storage already exists and its inactive words may be arbitrary. -/
abbrev Admission := Checked.Admission
abbrev EmptyAdmission := Checked.EmptyAdmission

#axiomsClean sparse_iff
#specNonCircular SparseChecks avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelContract
