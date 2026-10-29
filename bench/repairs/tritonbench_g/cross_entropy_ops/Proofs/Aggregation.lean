import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.RealGrid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Aggregation

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Aggregation
open VeriTile.Triton
open CrossEntropy1.Source (Memory Cell)
open RealSource (algebra Input)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def rankParameters (p : Parameters) (ranks rank : Nat) : Parameters :=
  {p with totalClasses:=ranks*p.cols,classStart:=rank*p.cols,split:=Bool.true}

def computed (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (rank : Nat) : Memory (WithBot ℝ) :=
  Grid.forward algebra (rankParameters p ranks rank) (memories rank) (CrossEntropy1.Math.blocks p.cols p.block)

def lossSlot (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (rank row block : Nat) : ℝ :=
  ((computed p ranks memories rank p.loss (p.lossBase+block*p.rows+row)).read .real).unbotD 0

def lseSlot (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (rank row block : Nat) : ℝ :=
  ((computed p ranks memories rank p.lse (p.lseBase+block*p.rows+row)).read .real).unbotD 0

/-- Real interpretation of the host's local-then-rank LSE operations, reading
actual grid output cells. This definition is not a Torch implementation proof. -/
def combinedLSE (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (row : Nat) : ℝ :=
  Real.log (∑ rank∈Finset.range ranks,
    Real.exp (Real.log (∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
      Real.exp (lseSlot p ranks memories rank row block))))

def combinedPartialLoss (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (row : Nat) : ℝ :=
  ∑ rank∈Finset.range ranks, ∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
    lossSlot p ranks memories rank row block

def combinedLoss (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (row : Nat) (target : Int) : ℝ :=
  if target=p.ignored then 0 else combinedPartialLoss p ranks memories row + combinedLSE p ranks memories row +
    if p.squareScale≠0 then p.squareScale*(combinedLSE p ranks memories row)^2 else 0

def combinedZLoss (p : Parameters) (ranks : Nat) (memories : Nat → Memory (WithBot ℝ)) (row : Nat) (target : Int) : ℝ :=
  if target=p.ignored then 0 else if p.squareScale≠0 then p.squareScale*(combinedLSE p ranks memories row)^2 else 0

theorem slots (p : Parameters) (ranks rank row block : Nat) (memories : Nat → Memory (WithBot ℝ))
    (sep : Grid.ForwardSeparation p) (rowBound : row<p.rows)
    (blockBound : block<CrossEntropy1.Math.blocks p.cols p.block) (positive : 0<p.block)
    (x : Nat → ℝ) (input : Input (rankParameters p ranks rank) (memories rank) row (fun j => x (rank*p.cols+j)))
    (target : Int) (labelRead : CrossEntropy1.Source.label p.toParameters (memories rank) row=target)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    lossSlot p ranks memories rank row block=
      CrossEntropy1.RealSource.partialLoss (rankParameters p ranks rank).toParameters
        (Spec.scaled p.logitScale (fun j => x (rank*p.cols+j))) target block ∧
    lseSlot p ranks memories rank row block=
      CrossEntropy1.Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j)*p.logitScale) := by
  let rp := rankParameters p ranks rank
  have separation : Grid.ForwardSeparation rp :=
    {outputDistinct:=sep.outputDistinct,lseLogits:=sep.lseLogits,lossLogits:=sep.lossLogits,
     lseLabels:=sep.lseLabels,lossLabels:=sep.lossLabels,zLoss:=sep.zLoss,zLse:=sep.zLse,
     zLogits:=sep.zLogits,zLabels:=sep.zLabels}
  have stores := Grid.forward_outputs algebra rp (memories rank) (CrossEntropy1.Math.blocks p.cols p.block)
    (row,block) ((CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨rowBound,blockBound⟩) separation
  have loss := RealSource.forward_split_loss rp (memories rank) row block (fun j => x (rank*p.cols+j))
    input target rfl noSmooth
  have lse := RealSource.forward_lse rp (memories rank) row block (fun j => x (rank*p.cols+j))
    input positive (CrossEntropy1.Math.block_starts p.cols p.block block positive blockBound)
  have labelRead' : CrossEntropy1.Source.label rp.toParameters (memories rank) row=target := labelRead
  constructor
  · change ((Grid.forward algebra rp (memories rank) _ rp.loss (rp.lossBase+block*rp.rows+row)).read .real).unbotD 0=_
    rw [stores.1,Cell.read_same,labelRead',loss,WithBot.unbotD_coe]
    rfl
  · change ((Grid.forward algebra rp (memories rank) _ rp.lse (rp.lseBase+block*rp.rows+row)).read .real).unbotD 0=_
    rw [stores.2.1,Cell.read_same,lse,WithBot.unbotD_coe]
    rfl

theorem actual_combined_lse (p : Parameters) (ranks row : Nat) (memories : Nat → Memory (WithBot ℝ))
    (sep : Grid.ForwardSeparation p) (rowBound : row<p.rows) (positive : 0<p.cols) (widthPositive : 0<p.block)
    (x : Nat → ℝ)
    (inputs : ∀ rank, rank<ranks → Input (rankParameters p ranks rank) (memories rank) row (fun j => x (rank*p.cols+j)))
    (target : Int) (labels : ∀ rank, rank<ranks → CrossEntropy1.Source.label p.toParameters (memories rank) row=target)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    combinedLSE p ranks memories row=Spec.lse (ranks*p.cols) p.logitScale x := by
  have each : ∀ rank∈Finset.range ranks, ∀ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
      lseSlot p ranks memories rank row block=
        CrossEntropy1.Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j)*p.logitScale) := by
    intro rank hr block hb
    exact (slots p ranks rank row block memories sep rowBound (Finset.mem_range.mp hb) widthPositive x
      (inputs rank (Finset.mem_range.mp hr)) target (labels rank (Finset.mem_range.mp hr)) noSmooth).2
  unfold combinedLSE
  have rewriteSlots : (∑ rank∈Finset.range ranks,
      Real.exp (Real.log (∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
        Real.exp (lseSlot p ranks memories rank row block))))=
      (∑ rank∈Finset.range ranks,
      Real.exp (Real.log (∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
        Real.exp (CrossEntropy1.Math.blockLSE p.cols p.block block (fun j => x (rank*p.cols+j)*p.logitScale))))) := by
    apply Finset.sum_congr rfl
    intro rank hr
    congr 2
    exact Finset.sum_congr rfl (fun block hb => congrArg Real.exp (each rank hr block hb))
  rw [rewriteSlots]
  exact CrossEntropy1.Aggregation.nested_lse ranks p.cols p.block positive widthPositive (Spec.scaled p.logitScale x)

/-- Both local splitting and distributed target ownership are derived from
actual loss cells. No rank-local expected result is supplied as a hypothesis. -/
theorem actual_combined_loss (p : Parameters) (ranks row : Nat) (memories : Nat → Memory (WithBot ℝ))
    (sep : Grid.ForwardSeparation p) (rowBound : row<p.rows) (positive : 0<p.cols) (widthPositive : 0<p.block)
    (x : Nat → ℝ)
    (inputs : ∀ rank, rank<ranks → Input (rankParameters p ranks rank) (memories rank) row (fun j => x (rank*p.cols+j)))
    (target : Int) (labels : ∀ rank, rank<ranks → CrossEntropy1.Source.label p.toParameters (memories rank) row=target)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (valid : target=p.ignored ∨ (0≤target ∧ target<(ranks*p.cols : Nat))) :
    combinedLoss p ranks memories row target=
      Spec.loss (ranks*p.cols) p.logitScale p.smoothing p.squareScale x target p.ignored ∧
    combinedZLoss p ranks memories row target=
      Spec.zLoss (ranks*p.cols) p.logitScale p.squareScale x target p.ignored := by
  have lse := actual_combined_lse p ranks row memories sep rowBound positive widthPositive x inputs target labels noSmooth
  constructor
  · by_cases ignored : target=p.ignored
    · simp [combinedLoss,Spec.loss,ignored]
    · have targetValid := valid.resolve_left ignored
      have partialEq : combinedPartialLoss p ranks memories row=
          ∑ rank∈Finset.range ranks, CrossEntropy1.Aggregation.rankContribution ranks p.cols target.toNat p.smoothing
            (Spec.scaled p.logitScale x) rank := by
        unfold combinedPartialLoss
        apply Finset.sum_congr rfl
        intro rank hr
        have each : (∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block), lossSlot p ranks memories rank row block)=
            ∑ block∈Finset.range (CrossEntropy1.Math.blocks p.cols p.block),
              CrossEntropy1.RealSource.partialLoss (rankParameters p ranks rank).toParameters
                (Spec.scaled p.logitScale (fun j => x (rank*p.cols+j))) target block := by
          apply Finset.sum_congr rfl
          intro block hb
          exact (slots p ranks rank row block memories sep rowBound (Finset.mem_range.mp hb) widthPositive x
            (inputs rank (Finset.mem_range.mp hr)) target (labels rank (Finset.mem_range.mp hr)) noSmooth).1
        rw [each]
        exact CrossEntropy1.Aggregation.actual_rank_contribution
          {p.toParameters with split:=Bool.true} ranks rank (Spec.scaled p.logitScale x) target targetValid.1 ignored widthPositive
      unfold combinedLoss
      rw [partialEq,lse]
      exact CrossEntropy1.Aggregation.global_loss ranks p.cols positive target p.ignored valid p.smoothing p.squareScale
        (Spec.scaled p.logitScale x)
  · unfold combinedZLoss Spec.zLoss
    rw [lse]
    by_cases ignored : target=p.ignored
    · simp [ignored]
    · simp only [ignored,if_false]
      by_cases z : p.squareScale=0
      · simp [z]
      · rw [if_pos z]
        ring

#axiomsClean slots
#axiomsClean actual_combined_lse
#axiomsClean actual_combined_loss
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Aggregation
