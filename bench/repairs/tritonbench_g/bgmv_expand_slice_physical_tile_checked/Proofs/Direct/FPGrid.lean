import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.FPExecution
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot

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

def run (M : Algebra α) (reduction : Reduction α) (h : Call) :
    List (Geometry.Program h) → Data α → Option (Data α)
  | [],data => some data
  | p::rest,data => do
      let state ← FP.Structural.exec (machine M reduction) (Kernels.source h) (entry h p.batch p.row data)
      run M reduction h rest (observed data state)

def fold (M : Algebra α) (reduction : Reduction α) (h : Call) : List (Geometry.Program h) → Data α → Data α
  | [],data => data
  | p::rest,data => fold M reduction h rest (updated M reduction h p data)

theorem one (M : Algebra α) (reduction : Reduction α) (h : Call) (primary : Kernels.Primary h)
    (p : Geometry.Program h) (data : Data α) (valid : data.indices p.batch=-1 ∨ 0 ≤ data.indices p.batch) :
    ∃ t,FP.Structural.exec (machine M reduction) (Kernels.source h) (entry h p.batch p.row data)=some t ∧
      observed data t=updated M reduction h p data := by
  obtain ⟨t,exec,memory⟩ := FPExecution.source_executes M reduction h primary p data valid
  refine ⟨t,exec,?_⟩
  unfold observed updated
  congr 1
  funext off
  by_cases sentinel : data.indices p.batch=-1
  · simp only [sentinel,if_true] at memory
    rw [memory]
    simp [sentinel,entry,Data.cells]
  · simp only [sentinel,if_false] at memory
    rw [memory]
    by_cases same : off=address h p
    all_goals
      simp only [address] at same
      simp [sentinel,entry,Data.cells,State.write,value,address,same]

theorem executes (M : Algebra α) (reduction : Reduction α) (h : Call) (primary : Kernels.Primary h)
    (ps : List (Geometry.Program h)) (data : Data α)
    (valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0 ≤ data.indices b) :
    run M reduction h ps data=some (fold M reduction h ps data) := by
  induction ps generalizing data with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨t,exec,read⟩ := one M reduction h primary p data (valid p.batch p.batchInside)
    simp only [run,exec,bind,Option.bind,read,fold]
    apply ih
    exact valid

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

#axiomsClean executes
#axiomsClean values
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPGrid
