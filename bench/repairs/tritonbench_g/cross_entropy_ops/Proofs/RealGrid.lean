import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Real
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Grid

namespace VeriTile.Bench.Repairs.CrossEntropyOps.RealGrid
open VeriTile.Triton
open CrossEntropy1.Source (Memory Cell)
open RealSource (algebra Input)
noncomputable section
set_option maxHeartbeats 6000000

theorem forward_region_frame (p : Parameters) (memory : Memory (WithBot ℝ))
    (count : Nat) (sep : Grid.ForwardSeparation p) (region : RegionName)
    (loss : region≠p.loss) (lse : region≠p.lse) (zloss : region≠p.zloss) (offset : Nat) :
    Grid.forward algebra p memory count region offset=memory region offset := by
  apply Grid.forward_complete_frame algebra p memory count sep
  intro q _ hw
  rcases hw with (hw|hw)|hw
  · exact loss hw.1
  · exact lse hw.1
  · exact zloss hw.2.1

/-- Actual source-grid stores, for every row, establish all three independent
Real results. This theorem does not assume a correct saved LSE. -/
theorem unsplit_outputs (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat)
    (rowBound : row<p.rows) (x : Nat → ℝ) (input : Input p memory row x)
    (sep : Grid.ForwardSeparation p) (positive : 0<p.cols) (covered : p.cols≤p.block)
    (unsplit : p.split=Bool.false) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (start : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : CrossEntropy1.Source.label p.toParameters memory row=p.ignored ∨
      (0≤CrossEntropy1.Source.label p.toParameters memory row ∧
        CrossEntropy1.Source.label p.toParameters memory row<(p.cols : Int))) :
    (Grid.forward algebra p memory 1 p.loss (p.lossBase+row)).read .real=
      ↑(Spec.loss p.cols p.logitScale p.smoothing p.squareScale x
        (CrossEntropy1.Source.label p.toParameters memory row) p.ignored) ∧
    (Grid.forward algebra p memory 1 p.lse (p.lseBase+row)).read .real=↑(Spec.lse p.cols p.logitScale x) ∧
    (Grid.forward algebra p memory 1 p.zloss (p.zlossBase+row)).read .real=
      ↑(Spec.zLoss p.cols p.logitScale p.squareScale x
        (CrossEntropy1.Source.label p.toParameters memory row) p.ignored) := by
  have stores := Grid.forward_outputs algebra p memory 1 (row,0)
    ((CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨rowBound,by omega⟩) sep
  simp only [Nat.zero_mul,Nat.add_zero] at stores
  have saved := RealSource.forward_lse p memory row 0 x input (by omega) (by simpa using positive)
  rw [CrossEntropy1.Math.block_zero_lse p.cols p.block covered] at saved
  change (Source.forwardValues algebra p memory row 0).lse=↑(Spec.lse p.cols p.logitScale x) at saved
  refine ⟨?_,?_,?_⟩
  · rw [stores.1,Cell.read_same]
    exact RealSource.forward_unsplit_loss p memory row x input _ positive covered unsplit noSmooth start total valid
  · rw [stores.2.1,Cell.read_same,saved]
  · rw [stores.2.2 unsplit,Cell.read_same,saved,RealSource.logging_zloss p _ _ unsplit]
    rfl

/-- A complete source-device forward/backward round trip, including backward
splitting and in-place output. The backward reads the LSE stored by this actual
forward grid. Host tensor preparation and native lowering are separate layers. -/
theorem unsplit_roundtrip_gradient (p : Parameters) (memory : Memory (WithBot ℝ))
    (x : Nat → Nat → ℝ) (upstream : Nat → ℝ)
    (inputs : ∀ row, row<p.rows → Input p memory row (x row))
    (upstreams : ∀ row, row<p.rows →
      (memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑(upstream row))
    (sep : Grid.ForwardSeparation p) (positive : 0<p.cols) (covered : p.cols≤p.block)
    (unsplit : p.split=Bool.false) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (start : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : ∀ row, row<p.rows → CrossEntropy1.Source.label p.toParameters memory row=p.ignored ∨
      (0≤CrossEntropy1.Source.label p.toParameters memory row ∧
        CrossEntropy1.Source.label p.toParameters memory row<(p.cols : Int)))
    (upLoss : p.upstream≠p.loss) (upLse : p.upstream≠p.lse) (upZloss : p.upstream≠p.zloss)
    (backWidth : Nat) (backPositive : 0<backWidth)
    (backSep : Grid.BackwardSeparation {p with block:=backWidth})
    (row col : Nat) (rowBound : row<p.rows) (colBound : col<p.cols) :
    (Grid.backward algebra {p with block:=backWidth} (Grid.forward algebra p memory 1)
      (CrossEntropy1.Math.blocks p.cols backWidth) p.gradient
        (CrossEntropy1.Source.gradientAddress p.toParameters row col)).read .real=
      ↑(Spec.gradient p.cols p.logitScale p.smoothing p.squareScale (upstream row) (x row)
        (CrossEntropy1.Source.label p.toParameters memory row) p.ignored col) := by
  let bp : Parameters := {p with block:=backWidth}
  let forwarded := Grid.forward algebra p memory 1
  have logitsFrame : ∀ i, forwarded p.logits i=memory p.logits i :=
    forward_region_frame p memory 1 sep p.logits sep.lossLogits.symm sep.lseLogits.symm sep.zLogits.symm
  have labelsFrame : ∀ i, forwarded (Region.cast p.labels) i=memory (Region.cast p.labels) i :=
    forward_region_frame p memory 1 sep (Region.cast p.labels) sep.lossLabels.symm sep.lseLabels.symm sep.zLabels.symm
  have upstreamFrame : ∀ i, forwarded p.upstream i=memory p.upstream i :=
    forward_region_frame p memory 1 sep p.upstream upLoss upLse upZloss
  have backInput : Input bp forwarded row (x row) := by
    intro j hj
    change (forwarded p.logits (CrossEntropy1.Source.logitAddress p.toParameters row j)).read .real=↑(x row j)
    rw [logitsFrame]
    exact inputs row rowBound j hj
  have backLabel : CrossEntropy1.Source.label bp.toParameters forwarded row=
      CrossEntropy1.Source.label p.toParameters memory row := by
    change (forwarded (Region.cast p.labels) (p.labelsBase+row)).read .int=_
    rw [labelsFrame]
    rfl
  have backUpstream : (forwarded bp.upstream (bp.upstreamBase+row*bp.upstreamStride)).read .real=↑(upstream row) := by
    change (forwarded p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=_
    rw [upstreamFrame]
    exact upstreams row rowBound
  have saved := (unsplit_outputs p memory row rowBound (x row) (inputs row rowBound) sep
    positive covered unsplit noSmooth start total (valid row rowBound)).2.1
  obtain ⟨q,hq,lane,rowEq,colEq⟩ := CrossEntropy1.Grid.owner bp.toParameters backPositive row col rowBound colBound
  have active : CrossEntropy1.Source.column bp.toParameters q.2 lane.val<bp.cols := by
    rw [colEq]
    exact colBound
  have stored := Grid.backward_output algebra bp forwarded (CrossEntropy1.Math.blocks bp.cols bp.block)
    q hq backSep lane active
  have laneValue := RealSource.backward_lane bp forwarded row q.2 (x row) backInput
    (CrossEntropy1.Source.label p.toParameters memory row) (Spec.lse p.cols p.logitScale (x row)) (upstream row)
    backLabel saved backUpstream sep.lseLogits upLse.symm noSmooth lane active
  rw [colEq] at laneValue
  rw [rowEq,colEq] at stored
  change (Grid.backward algebra bp forwarded (CrossEntropy1.Math.blocks bp.cols bp.block) bp.gradient
    (CrossEntropy1.Source.gradientAddress bp.toParameters row col)).read .real=_
  rw [stored,Cell.read_same,laneValue]
  simp only [bp,Spec.gradient,start,total,Nat.cast_zero,sub_zero]

#axiomsClean forward_region_frame
#axiomsClean unsplit_outputs
#axiomsClean unsplit_roundtrip_gradient
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.RealGrid
