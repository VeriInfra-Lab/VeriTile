import bench.tritonbench_g.lora_expand_gemv.Proofs.Contract

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Contract
noncomputable section
open Classical
abbrev TensorInfo := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.TensorInfo
abbrev rows := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows
abbrev columns := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns
abbrev rank := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank
abbrev width := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width
abbrev Dense := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Dense
abbrev Singleton := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Singleton
abbrev Inputs := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Inputs
abbrev Scalars := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Scalars
abbrev tensors := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.tensors
abbrev reads := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.reads
abbrev block := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.block
abbrev geometry := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry
abbrev Layouts := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Layouts
abbrev DTypes := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.DTypes
abbrev Ranks := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Ranks
abbrev Devices := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Devices
abbrev Shapes := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Shapes
abbrev Limits := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Limits
abbrev Aligned := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Aligned
abbrev Metadata := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.Metadata

/-- Observations of complete untyped_storage(), not merely the Tensor view.
No identity-disjointness rule is assumed. Zero-byte ranges are empty. -/
structure StorageRange where
  base : Nat
  bytes : Nat
  deriving DecidableEq, Repr
structure Backings where
  x : StorageRange
  w : StorageRange
  y : StorageRange
  indices : StorageRange
  deriving DecidableEq, Repr

def StorageRange.covers (a : StorageRange) (address : Nat) :=
  a.base≤address ∧ address<a.base+a.bytes

def StorageRange.disjoint (a b : StorageRange) :=
  a.bytes=0 ∨ b.bytes=0 ∨ a.base+a.bytes≤b.base ∨ b.base+b.bytes≤a.base

def Separate (a : Backings) : Prop :=
  a.y.disjoint a.x ∧ a.y.disjoint a.w ∧ a.y.disjoint a.indices

theorem disjoint_byte (a b : StorageRange) (h : a.disjoint b)
    (address : Nat) (ha : a.covers address) (hb : b.covers address) : False := by
  rcases h with h|h|h|h <;> simp only [StorageRange.covers] at ha hb <;> omega

theorem disjoint_iff_no_common_byte (a b : StorageRange) :
    a.disjoint b ↔ ∀ address,¬(a.covers address ∧ b.covers address) := by
  constructor
  · intro h address both; exact disjoint_byte a b h address both.1 both.2
  · intro h
    by_contra hn
    simp only [StorageRange.disjoint,not_or] at hn
    have both : a.covers (max a.base b.base) ∧ b.covers (max a.base b.base) := by
      simp only [StorageRange.covers]
      omega
    exact h _ both

structure InputContract (t : Inputs) (s : Scalars) (a : Backings) (indices : Nat→Int) : Prop where
  layouts : Layouts t
  dtypes : DTypes t
  ranks : Ranks t
  singleton : Singleton t.w
  devices : Devices t
  dense : ∀ tensor∈tensors t,Dense tensor
  scalarType : s.addBool=Bool.true
  shapes : Shapes t
  limits : Limits t
  separate : Separate a
  aligned : Aligned t
  metadata : Metadata t indices

def pythonChecks (t : Inputs) (s : Scalars) (a : Backings) (indices : Nat→Int) : List Bool :=
  [decide (Layouts t),decide (DTypes t),decide (Ranks t),decide (Singleton t.w),decide (Devices t),
   decide (∀ tensor∈tensors t,Dense tensor),s.addBool,decide (Shapes t),decide (Limits t),
   decide (Separate a),decide (Aligned t),decide (Metadata t indices)]
def Admitted (t : Inputs) (s : Scalars) (a : Backings) (indices : Nat→Int) := (pythonChecks t s a indices).all id=Bool.true

theorem python_checks_iff (t : Inputs) (s : Scalars) (a : Backings) (indices : Nat→Int) :
    Admitted t s a indices ↔ InputContract t s a indices := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨ha,hb,hc,hd,he,hf,hg,hh,hi,hj,hk,hl⟩;exact ⟨ha,hb,hc,hd,he,hf,hg,hh,hi,hj,hk,hl⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scalarType,h.shapes,h.limits,h.separate,h.aligned,h.metadata⟩

#axiomsClean disjoint_byte
#axiomsClean disjoint_iff_no_common_byte
#axiomsClean python_checks_iff
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Contract
