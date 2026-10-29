import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Aggregation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardGrid

namespace VeriTile.Bench.Repairs.CrossEntropy1.Composition
open VeriTile.Triton FP.Structural Source
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- The actual resulting memory of each selected rank-local source grid. -/
def rankForward (p : Parameters) (ranks rank : Nat) (memory : Memory (WithBot ℝ)) : Memory (WithBot ℝ) :=
  Grid.forward RealSource.algebra (Aggregation.rankParameters p ranks rank) memory (Math.blocks p.cols p.block)

def lossSlot (p : Parameters) (ranks rank : Nat) (memory : Memory (WithBot ℝ)) (row block : Nat) : ℝ :=
  WithBot.unbotD 0 ((rankForward p ranks rank memory p.loss (p.lossBase+block*p.rows+row)).read .real)

def lseSlot (p : Parameters) (ranks rank : Nat) (memory : Memory (WithBot ℝ)) (row block : Nat) : ℝ :=
  WithBot.unbotD 0 ((rankForward p ranks rank memory p.lse (p.lseBase+block*p.rows+row)).read .real)

def savedGlobalLSE (p : Parameters) (ranks : Nat) (memory : Nat → Memory (WithBot ℝ)) (row : Nat) : ℝ :=
  Real.log (∑ rank∈Finset.range ranks, Real.exp (Real.log
    (∑ block∈Finset.range (Math.blocks p.cols p.block), Real.exp (lseSlot p ranks rank (memory rank) row block))))

def splitLoss (p : Parameters) (ranks : Nat) (memory : Nat → Memory (WithBot ℝ)) (row : Nat) (target : Int) : ℝ :=
  if target=p.ignored then 0 else
    (∑ rank∈Finset.range ranks, ∑ block∈Finset.range (Math.blocks p.cols p.block),
      lossSlot p ranks rank (memory rank) row block)+savedGlobalLSE p ranks memory row+
    (if p.squareScale≠0 then p.squareScale*(savedGlobalLSE p ranks memory row)^2 else 0)

theorem lse_slot (p : Parameters) (ranks rank row block : Nat) (memory : Memory (WithBot ℝ))
    (x : Nat → ℝ) (positive : 0<p.block) (hr : row<p.rows) (hb : block<Math.blocks p.cols p.block)
    (separation : Grid.ForwardSeparation (Aggregation.rankParameters p ranks rank))
    (input : RealSource.Input (Aggregation.rankParameters p ranks rank) memory row (fun j => x (rank*p.cols+j))) :
    lseSlot p ranks rank memory row block=
      Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j)) := by
  have outputs := Grid.forward_outputs RealSource.algebra (Aggregation.rankParameters p ranks rank) memory
    (Math.blocks p.cols p.block) (row,block) ((Grid.task_mem _ _ _).mpr ⟨hr,hb⟩) separation
  have finite := RealSource.forward_lse (Aggregation.rankParameters p ranks rank) memory row block
    (fun j => x (rank*p.cols+j)) input positive (Math.block_starts p.cols p.block block positive hb)
  have stored : Grid.forward RealSource.algebra (Aggregation.rankParameters p ranks rank) memory
      (Math.blocks p.cols p.block) p.lse (p.lseBase+block*p.rows+row)=
        .mk .real (forwardValues RealSource.algebra (Aggregation.rankParameters p ranks rank) memory row block).lse := outputs.2
  unfold lseSlot rankForward
  rw [stored,Source.Cell.read_same,finite]
  rfl

theorem loss_slot (p : Parameters) (ranks rank row block : Nat) (memory : Memory (WithBot ℝ))
    (x : Nat → ℝ) (target : Int) (hr : row<p.rows) (hb : block<Math.blocks p.cols p.block)
    (separation : Grid.ForwardSeparation (Aggregation.rankParameters p ranks rank))
    (input : RealSource.Input (Aggregation.rankParameters p ranks rank) memory row (fun j => x (rank*p.cols+j)))
    (label : Source.label (Aggregation.rankParameters p ranks rank) memory row=target)
    (split : p.split=Bool.true) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    lossSlot p ranks rank memory row block=
      RealSource.partialLoss (Aggregation.rankParameters p ranks rank) (fun j => x (rank*p.cols+j)) target block := by
  have outputs := Grid.forward_outputs RealSource.algebra (Aggregation.rankParameters p ranks rank) memory
    (Math.blocks p.cols p.block) (row,block) ((Grid.task_mem _ _ _).mpr ⟨hr,hb⟩) separation
  have finite := RealSource.split_loss (Aggregation.rankParameters p ranks rank) memory row block
    (fun j => x (rank*p.cols+j)) input target split noSmooth
  have stored : Grid.forward RealSource.algebra (Aggregation.rankParameters p ranks rank) memory
      (Math.blocks p.cols p.block) p.loss (p.lossBase+block*p.rows+row)=
        .mk .real (forwardLoss RealSource.algebra (Aggregation.rankParameters p ranks rank) memory row block
          (Source.label (Aggregation.rankParameters p ranks rank) memory row)
          (forwardValues RealSource.algebra (Aggregation.rankParameters p ranks rank) memory row block)) := outputs.1
  unfold lossSlot rankForward
  rw [stored,Source.Cell.read_same,label,finite]
  rfl

/-- Global LSE is derived from actual source-grid output stores on all ranks,
not supplied as an output premise. This is the value subsequently saved. -/
theorem saved_lse (p : Parameters) (ranks row : Nat) (memory : Nat → Memory (WithBot ℝ))
    (x : Nat → ℝ) (classesPositive : 0<p.cols) (positive : 0<p.block) (hr : row<p.rows)
    (separation : ∀ rank, rank<ranks → Grid.ForwardSeparation (Aggregation.rankParameters p ranks rank))
    (input : ∀ rank, rank<ranks → RealSource.Input (Aggregation.rankParameters p ranks rank)
      (memory rank) row (fun j => x (rank*p.cols+j))) :
    savedGlobalLSE p ranks memory row=Math.lse (ranks*p.cols) x := by
  unfold savedGlobalLSE
  have slots : ∀ rank∈Finset.range ranks, ∀ block∈Finset.range (Math.blocks p.cols p.block),
      lseSlot p ranks rank (memory rank) row block=
        Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j)) := by
    intro rank hrank block hblock
    exact lse_slot p ranks rank row block (memory rank) x positive hr (Finset.mem_range.mp hblock)
      (separation rank (Finset.mem_range.mp hrank)) (input rank (Finset.mem_range.mp hrank))
  have outer : (∑ rank∈Finset.range ranks, Real.exp (Real.log
      (∑ block∈Finset.range (Math.blocks p.cols p.block), Real.exp (lseSlot p ranks rank (memory rank) row block))))=
      ∑ rank∈Finset.range ranks, Real.exp (Real.log
        (∑ block∈Finset.range (Math.blocks p.cols p.block),
          Real.exp (Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j))))) := by
    apply Finset.sum_congr rfl
    intro rank hrank
    congr 2
    apply Finset.sum_congr rfl
    intro block hblock
    rw [slots rank hrank block hblock]
  rw [outer]
  exact Aggregation.nested_lse ranks p.cols p.block classesPositive positive x

/-- All split losses, including rank ownership, smoothing, z-loss and ignored
rows, reduce to the independent full global cross-entropy formula. -/
theorem split_loss (p : Parameters) (ranks row : Nat) (memory : Nat → Memory (WithBot ℝ))
    (x : Nat → ℝ) (target : Int) (classesPositive : 0<p.cols) (positive : 0<p.block) (hr : row<p.rows)
    (separation : ∀ rank, rank<ranks → Grid.ForwardSeparation (Aggregation.rankParameters p ranks rank))
    (input : ∀ rank, rank<ranks → RealSource.Input (Aggregation.rankParameters p ranks rank)
      (memory rank) row (fun j => x (rank*p.cols+j)))
    (labels : ∀ rank, rank<ranks → Source.label (Aggregation.rankParameters p ranks rank) (memory rank) row=target)
    (split : p.split=Bool.true) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (valid : target=p.ignored ∨ (0≤target ∧ target<(ranks*p.cols : Nat))) :
    splitLoss p ranks memory row target=Math.loss (ranks*p.cols) x target p.ignored p.smoothing p.squareScale := by
  by_cases ignored : target=p.ignored
  · simp [splitLoss,Math.loss,ignored]
  · have targetValid := valid.resolve_left ignored
    have rankSum : ∀ rank∈Finset.range ranks,
        (∑ block∈Finset.range (Math.blocks p.cols p.block), lossSlot p ranks rank (memory rank) row block)=
          Aggregation.rankContribution ranks p.cols target.toNat p.smoothing x rank := by
      intro rank hrank
      have perBlock : ∀ block∈Finset.range (Math.blocks p.cols p.block),
          lossSlot p ranks rank (memory rank) row block=
            RealSource.partialLoss (Aggregation.rankParameters p ranks rank) (fun j => x (rank*p.cols+j)) target block := by
        intro block hblock
        exact loss_slot p ranks rank row block (memory rank) x target hr (Finset.mem_range.mp hblock)
          (separation rank (Finset.mem_range.mp hrank)) (input rank (Finset.mem_range.mp hrank))
          (labels rank (Finset.mem_range.mp hrank)) split noSmooth
      rw [Finset.sum_congr rfl perBlock]
      exact Aggregation.actual_rank_contribution p ranks rank x target targetValid.1 ignored positive
    unfold splitLoss
    rw [Finset.sum_congr rfl rankSum,saved_lse p ranks row memory x classesPositive positive hr separation input]
    exact Aggregation.global_loss ranks p.cols classesPositive target p.ignored valid p.smoothing p.squareScale x

#axiomsClean saved_lse
#axiomsClean split_loss
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Composition
