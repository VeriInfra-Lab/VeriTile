import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeControl

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.PhysicalGrid
open VeriTile.Triton FP.Structural FPValues
noncomputable section
open Classical
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- Actual output word address, using the same padded allocation consumed by
the source and native-index proof. A local store changes one encoded word. -/
def address (h : Call) (storage : Storage.Placement h) (p : Geometry.Program h) : Nat :=
  storage.output.address (FPGrid.address h p)

def step (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h)
    (p : Geometry.Program h) (data : Data α) (heap : Nat → α) : Nat → α :=
  if data.indices p.batch=-1 then heap else
    Function.update heap (address h storage p) (FPGrid.value M reduction h p data)

def run (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h) :
    List (Geometry.Program h) → Data α → (Nat → α) → (Nat → α)
  | [],_,heap => heap
  | p::rest,data,heap => run M reduction h storage rest
      (FPGrid.updated M reduction h p data) (step M reduction h storage p data heap)

theorem step_agrees (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h)
    (p : Geometry.Program h) (data : Data α) (heap : Nat → α)
    (entry : ∀ off,off<storage.output.extent → heap (storage.output.address off)=data.output off) :
    ∀ off,off<storage.output.extent →
      step M reduction h storage p data heap (storage.output.address off)=
        (FPGrid.updated M reduction h p data).output off := by
  intro off inside
  by_cases sentinel : data.indices p.batch=-1
  · simp [step,FPGrid.updated,sentinel,entry off inside]
  · by_cases same : off=FPGrid.address h p
    · subst off
      simp [step,FPGrid.updated,sentinel,address]
    · have different : storage.output.address off≠address h storage p := by
        intro equal
        exact same (Storage.address_injective storage.output _ _ equal)
      simp [step,FPGrid.updated,sentinel,same,Function.update_of_ne different,entry off inside]

theorem agrees (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h)
    (ps : List (Geometry.Program h)) (data : Data α) (heap : Nat → α)
    (entry : ∀ off,off<storage.output.extent → heap (storage.output.address off)=data.output off) :
    ∀ off,off<storage.output.extent →
      run M reduction h storage ps data heap (storage.output.address off)=
        (FPGrid.fold M reduction h ps data).output off := by
  induction ps generalizing data heap with
  | nil => exact entry
  | cons p rest ih =>
    exact ih _ _ (step_agrees M reduction h storage p data heap entry)

theorem frame (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h)
    (ps : List (Geometry.Program h)) (data : Data α) (heap : Nat → α) (observed : Nat)
    (outside : ∀ p∈ps,observed≠address h storage p) :
    run M reduction h storage ps data heap observed=heap observed := by
  induction ps generalizing data heap with
  | nil => rfl
  | cons p rest ih =>
    rw [run,ih _ _ (fun q hq => outside q (by simp [hq]))]
    by_cases sentinel : data.indices p.batch=-1
    all_goals simp [step,sentinel,Function.update_of_ne (outside p (by simp))]

/-- Source-program output is read back through the actual native scratch
address before the host's selected-slice copy. -/
theorem value (M : Algebra α) (reduction : Reduction α) (h : Call) (storage : Storage.Placement h)
    (schedule : Snapshot.Schedule h) (data : Data α) (heap : Nat → α)
    (entry : ∀ off,off<storage.output.extent → heap (storage.output.address off)=data.output off)
    (p : Geometry.Program h) (active : data.indices p.batch≠-1) :
    run M reduction h storage schedule.programs data heap (address h storage p)=FPGrid.value M reduction h p data := by
  rw [address,agrees M reduction h storage schedule.programs data heap entry]
  · exact FPGrid.values M reduction h schedule.programs (by simpa [Geometry.outputOffset,FPGrid.address] using schedule.unique)
      data p (schedule.covers p) active
  · rw [storage.outputSize]
    simpa [Geometry.outputOffset,FPGrid.address] using Geometry.output_inside h p ⟨0,by decide⟩

#axiomsClean step_agrees
#axiomsClean agrees
#axiomsClean value
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.PhysicalGrid
