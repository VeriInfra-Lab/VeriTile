import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.RealGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ViewGuard

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.RealHost
open VeriTile.Triton Snapshot
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

def initial (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0) : BlockState where
  mem := fun r off => if r="X" then .real (packed.input off)
    else if r="W" then .real (packed.weights off)
    else if r="Y" then .real (packed.output off)
    else if r="Indices" then .of .int (packed.indices off) else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0

/-- Independent mathematical caller-coordinate GEMV, not the source evaluator. -/
def math (h : Call) (logical : Logical ℝ) (batch row slot : Nat) : ℝ :=
  let dot := ∑ k : Fin h.K,logical.input batch k.val*logical.weight slot row k.val
  if h.add then dot+logical.output batch (h.slice+row) else dot

theorem formula_snapshot (h : Call) (logical : Logical ℝ) (packed : Packed ℝ h logical 0)
    (p : Geometry.Program h) (slot : Nat) (selected : slot<h.adapters) :
    RealExecution.formula h p (initial h logical packed) slot=math h logical p.batch p.row slot := by
  have col : h.slice+p.row<h.columns := by have := h.sliceInside; have := p.rowInside; omega
  have out := packed.outputCopy p.batch (h.slice+p.row) p.batchInside col
  have out' : packed.output (Geometry.outputOffset h p.batch p.row 0)=logical.output p.batch (h.slice+p.row) := by
    simpa [Geometry.outputOffset,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using out
  have dot : (∑ k : Fin h.K,packed.input (p.batch*h.BK+k.val)*
      packed.weights (slot*((h.N+255)*h.BK)+p.row*h.BK+k.val))=
      ∑ k : Fin h.K,logical.input p.batch k.val*logical.weight slot p.row k.val := by
    apply Finset.sum_congr rfl
    intro k _
    rw [packed.inputCopy p.batch k.val p.batchInside k.isLt,
      packed.weightCopy slot p.row k.val selected p.rowInside k.isLt]
  simp [RealExecution.formula,math,initial,BlockState.readMem,MemCell.real,MemCell.of,MemCell.readAs,out',dot]

theorem complete (h : Call) (schedule : Schedule h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0 ≤ logical.indices b ∧ (logical.indices b).toNat<h.adapters)) :
    ∃ t,RealGrid.run h schedule.programs (initial h logical packed)=some t ∧
      (∀ p : Geometry.Program h,logical.indices p.batch≠-1 →
        t.readMem "Y" (Geometry.outputOffset h p.batch p.row 0)=
          math h logical p.batch p.row (logical.indices p.batch).toNat) ∧
      (∀ r off,¬RealGrid.Writes h logical.indices schedule.programs r off →
        t.mem r off=(initial h logical packed).mem r off) := by
  obtain ⟨t,run,values,frame⟩ := RealGrid.complete h schedule.programs schedule.unique logical.indices
    (fun b hb => (valid b hb).imp_right And.left) (initial h logical packed) (by
      intro b hb
      simp [initial,BlockState.readMemValue,BlockState.readMemTyped,packed.metadataCopy b hb])
  refine ⟨t,run,?_,frame⟩
  intro p active
  rw [values p (schedule.covers p) active]
  exact formula_snapshot h logical packed p (logical.indices p.batch).toNat
    ((valid p.batch p.batchInside).resolve_left active).2

#axiomsClean formula_snapshot
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.RealHost
