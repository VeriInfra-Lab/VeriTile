import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedExecution
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.FPGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Contract
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Grid
open VeriTile Triton FP.Structural Direct FPValues
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

def run (M : Algebra α) (reduction : Reduction α) (h : Call) :
    List (Geometry.Program h) → Data α → Option (Data α)
  | [],data => some data
  | p::rest,data => do
      let t ← FP.Structural.exec (machine M reduction) (Surface.source h) (entry h p.batch p.row data)
      run M reduction h rest (FPGrid.observed data t)

theorem one (M : Algebra α) (reduction : Reduction α) (h : Call)
    (p : Geometry.Program h) (data : Data α) (valid : data.indices p.batch=-1 ∨ 0≤data.indices p.batch) :
    ∃ t,FP.Structural.exec (machine M reduction) (Surface.source h) (entry h p.batch p.row data)=some t ∧
      FPGrid.observed data t=FPGrid.updated M reduction h p data := by
  obtain ⟨t,exec,memory⟩ := Execution.source_executes M reduction h p data valid
  refine ⟨t,exec,?_⟩
  unfold FPGrid.observed FPGrid.updated
  congr 1
  funext off
  by_cases sentinel : data.indices p.batch=-1
  · simp only [sentinel,if_true] at memory
    rw [memory]
    simp [sentinel,entry,Data.cells]
  · simp only [sentinel,if_false] at memory
    rw [memory]
    by_cases same : off=FPGrid.address h p
    all_goals
      simp only [FPGrid.address] at same
      simp [sentinel,entry,Data.cells,State.write,FPGrid.value,FPGrid.address,same]

theorem executes (M : Algebra α) (reduction : Reduction α) (h : Call)
    (ps : List (Geometry.Program h)) (data : Data α)
    (valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b) :
    run M reduction h ps data=some (FPGrid.fold M reduction h ps data) := by
  induction ps generalizing data with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨t,exec,read⟩ := one M reduction h p data (valid p.batch p.batchInside)
    simp only [run,exec,bind,Option.bind,read,FPGrid.fold]
    exact ih _ valid

theorem complete (x w y i : Contract.TensorInfo) (s : Contract.Scalars) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i s indices) (M : Algebra α) (reduction : Reduction α)
    (data : Data α) (metadata : ∀ b,b< i.numel → data.indices b=indices b) :
    let h:=Contract.call x w y i s indices ok
    run M reduction h (Geometry.programs h) data=some (FPGrid.fold M reduction h (Geometry.programs h) data) ∧
      (∀ p : Geometry.Program h,indices p.batch≠-1 →
        (FPGrid.fold M reduction h (Geometry.programs h) data).output (FPGrid.address h p)=
          FPGrid.value M reduction h p data) := by
  dsimp only
  let h:=Contract.call x w y i s indices ok
  by_cases empty : h.N=0
  · have noProgram : Geometry.programs h=[] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro p _
      have := p.rowInside
      omega
    refine ⟨?_,?_⟩
    · change run M reduction h (Geometry.programs h) data=some (FPGrid.fold M reduction h (Geometry.programs h) data)
      rw [noProgram];rfl
    · intro p
      have inside : p.row<h.N := p.rowInside
      omega
  · have valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b := by
      intro b hb
      rw [metadata b hb]
      exact (Contract.selected_valid x w y i s indices ok b 0 hb (by change 0<h.N;omega)).imp_right And.left
    refine ⟨executes M reduction h _ data valid,?_⟩
    intro p active
    apply FPGrid.values M reduction h _ _ data p (Geometry.programs_complete h p)
    · rw [metadata p.batch p.batchInside];exact active
    · simpa [FPGrid.address,Geometry.outputOffset,VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry.outputOffset] using Geometry.programs_unique h

#axiomsClean executes
#axiomsClean complete
#specNonCircular FPGrid.value avoiding [run,FP.Structural.exec,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Grid
