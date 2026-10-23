import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Host
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Numerics
open VeriTile.Triton Logical
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent binary32 addition, round-to-nearest-even and gradual
input/result underflow. The raw source theorem applies beyond this profile. -/
@[irreducible] def referenceAdd (x y : Word) : Word := FP.add .ieeeValue x y

def finite (w : Word) : Prop := w.toRat?.isSome
def LocalAddAgreement (actual : Word → Word → Word) : Prop :=
  ∀ x y,finite x → finite y → actual x y=referenceAdd x y

/-- Domain at every actual addition, including the accumulated operand.
Finite initial values alone do not exclude intermediate overflow. -/
def PairsDomain {A : Type} {R N C : Nat} (predicate : A → Prop) (actual : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) :
    List (Fin N) → (Coord R C → A) → Prop
  | [],_ => True
  | i::rest,g => (∀ c,predicate (g (index i,c)) ∧ predicate (h (i,c))) ∧
      PairsDomain predicate actual index h rest (row actual index h i g)

theorem rows_equiv_generic {A : Type} {R N C : Nat} (predicate : A → Prop) (actual reference : A → A → A)
    (localAdd : ∀ x y,predicate x → predicate y → actual x y=reference x y)
    (index : Fin N → Fin R) (h : Coord N C → A) (is : List (Fin N)) (g : Coord R C → A)
    (domain : PairsDomain predicate actual index h is g) :
    rows actual index h is g=rows reference index h is g := by
  induction is generalizing g with
  | nil => rfl
  | cons i rest ih =>
    have step : row actual index h i g=row reference index h i g := by
      funext q
      by_cases same : index i=q.1
      · simp only [row,same,if_pos]
        apply localAdd
        · simpa only [same] using (domain.1 q.2).1
        · exact (domain.1 q.2).2
      · simp [row,same]
    change rows actual index h rest (row actual index h i g)=_
    rw [ih _ domain.2,step]
    rfl

def Domain (r : Request) (guard : InputContract r) (actual : Word → Word → Word)
    (g h : Nat → Word) : Prop :=
  PairsDomain (C:=r.C) finite actual (Native.selected r guard) (fun q => h (Host.flat q))
    (List.finRange r.N) (fun q => g (Host.flat q))

theorem source_reference (rest : FP.Structural.Algebra Word) (actual : Word → Word → Word)
    (localAdd : LocalAddAgreement actual) (r : Request) (guard : InputContract r)
    (g h : Nat → Word) (domain : Domain r guard actual g h) (object : Nat)
    (device : VeriTile.Torch.Device) (q : Coord r.R r.C) :
    (Host.run (Host.backend rest actual) r guard g h object device).destination (Host.flat q)=
      Host.orderedTarget referenceAdd r g h q := by
  rw [Host.source_readback]
  change rows actual _ _ _ _ q=_
  rw [rows_equiv_generic finite actual referenceAdd localAdd _ _ _ _ domain]
  exact (Host.source_readback (Host.backend rest referenceAdd) r guard g h object device q).symm.trans
    (Host.ordered_readback (Host.backend rest referenceAdd) r guard g h object device q)

#axiomsClean rows_equiv_generic
#axiomsClean source_reference
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Numerics
