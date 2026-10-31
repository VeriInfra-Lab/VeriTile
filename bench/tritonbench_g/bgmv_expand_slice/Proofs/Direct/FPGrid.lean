import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPReadback
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Geometry
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPControl
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPGrid
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

abbrev address (h : Call) (p : Geometry.Program h) := p.batch*h.columns+p.row+h.slice

def value (M : Algebra α) (reduction : Reduction α) (h : Call) (p : Geometry.Program h) (data : Data α) : α :=
  result M reduction h p.batch (data.indices p.batch).toNat p.row data ⟨0,by decide⟩

def updated (M : Algebra α) (reduction : Reduction α) (h : Call) (p : Geometry.Program h) (data : Data α) : Data α :=
  {data with output := fun off => if data.indices p.batch≠-1 ∧ off=address h p then
      FPGrid.value M reduction h p data else data.output off}

def observed (data : Data α) (state : State α) : Data α :=
  {data with output := fun off => (state.mem "Y" off).read .real}

def fold (M : Algebra α) (reduction : Reduction α) (h : Call) : List (Geometry.Program h) → Data α → Data α
  | [],data => data
  | p::rest,data => fold M reduction h rest (updated M reduction h p data)

theorem value_after_other (M : Algebra α) (reduction : Reduction α) (h : Call)
    (p q : Geometry.Program h) (data : Data α) (different : address h q≠address h p) :
    value M reduction h q (updated M reduction h p data)=value M reduction h q data := by
  unfold value result sums products loaded weight old updated
  simp [address] at different
  simp [different]

theorem frame (M : Algebra α) (reduction : Reduction α) (h : Call)
    (ps : List (Geometry.Program h)) (data : Data α) (off : Nat)
    (outside : ∀ p∈ps,off≠address h p) :
    (fold M reduction h ps data).output off=data.output off := by
  induction ps generalizing data with
  | nil => rfl
  | cons p rest ih =>
    rw [fold,ih _ (fun q hq => outside q (by simp [hq]))]
    simp [updated,outside p (by simp)]

theorem values (M : Algebra α) (reduction : Reduction α) (h : Call)
    (ps : List (Geometry.Program h)) (unique : ps.Pairwise (fun p q => address h p≠address h q))
    (data : Data α) (p : Geometry.Program h) (member : p∈ps) (active : data.indices p.batch≠-1) :
    (fold M reduction h ps data).output (address h p)=value M reduction h p data := by
  induction ps generalizing data with
  | nil => simp at member
  | cons q rest ih =>
    obtain ⟨different,uniqueRest⟩ := List.pairwise_cons.mp unique
    rcases List.mem_cons.mp member with same | later
    · subst q
      rw [fold,frame M reduction h rest _ _ (fun q hq => different q hq)]
      simp [updated,active]
    · rw [fold,ih uniqueRest (updated M reduction h q data) later active]
      exact value_after_other M reduction h q p data (Ne.symm (different p later))

#axiomsClean values
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPGrid
