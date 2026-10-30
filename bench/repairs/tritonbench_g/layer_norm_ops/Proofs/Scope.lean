import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage
namespace VeriTile.Bench.Repairs.LayerNormOps.Scope
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

/-- Ordinary contiguous default-FP32 wrapper contracts. Optional-pointer
strides are the actual wrapper arguments: zero when absent, including Mean
in RMS mode. Numeric-statistic or kernel-result equations are deliberately
absent. The host binds launch width to ceil(M/sm_count), after its bounded
ordinary arithmetic bridge. -/
structure Supported (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat) : Prop where
  columns : 0 < fp.columns
  columnsLimit : fp.columns ≤ 16384
  rowsBound : rows*fp.columns < 2147483648
  programsPositive : 0 < programs
  partialBound : programs*fp.columns < 2147483648
  sameColumns : bp.columns=fp.columns
  sameRows : bp.rows=rows
  sameBlock : bp.block=fp.block
  blockCovers : fp.columns ≤ fp.block
  blockLimit : fp.block ≤ 16384
  powerOfTwo : ∃ k, fp.block=2^k
  forwardStrides : fp.xStride=fp.columns ∧ fp.yStride=fp.columns ∧
    fp.residualStride=(if fp.hasResidual then fp.columns else 0) ∧
    fp.residualOutStride=(if fp.storeResidual then fp.columns else 0)
  backwardStrides : bp.xStride=bp.columns ∧
    bp.yStride=(if bp.recompute then bp.columns else 0) ∧ bp.dyStride=bp.columns ∧
    bp.dxStride=bp.columns ∧ bp.dresStride=(if bp.hasDres then bp.columns else 0)
  width : bp.rowsPerProgram=Launch.rowsPerProgram rows programs
  sameRms : bp.rms=fp.rms
  sameBias : bp.hasBias=fp.hasBias
  sameWeight : bp.weight=fp.weight
  sameBiasRegion : bp.bias=fp.bias
  storeResidual : fp.storeResidual=fp.hasResidual
  noSeparateDres : bp.storeDres=Bool.false
  savedZ : bp.x=if fp.storeResidual then fp.residualOut else fp.x
  savedMean : bp.mean=fp.mean
  savedRstd : bp.rstd=fp.rstd
  savedStrides : bp.meanStride=(if bp.rms then 0 else 1) ∧ bp.rstdStride=1
  /-- Include padded lanes and the rounded backward row groups: idle CTAs
  still form row_start pointers and the end expression before the row loop. -/
  sourceIndexBounds : rows*fp.columns+fp.block < 2147483648 ∧
    programs*bp.rowsPerProgram*fp.columns+fp.block < 2147483648 ∧
    programs*fp.columns+fp.block < 2147483648

/-- Forward pointers include every padded lane of the selected source block. -/
theorem forward_index (fp : ForwardParameters) (bp : BackwardParameters) (rows programs row lane : Nat)
    (h : Supported fp bp rows programs) (hr : row<rows) (hl : lane<fp.block) :
    row*fp.columns+lane < 2147483648 := by
  have hm := Nat.mul_le_mul_right fp.columns (Nat.le_of_lt hr)
  exact lt_of_le_of_lt (Nat.add_le_add hm (Nat.le_of_lt hl)) h.sourceIndexBounds.1

/-- Both the exclusive loop end and initial row pointers are represented,
including idle programs. This uses the rounded launch extent, not just M. -/
theorem backward_program_indices (fp : ForwardParameters) (bp : BackwardParameters)
    (rows programs pid lane : Nat) (h : Supported fp bp rows programs)
    (hp : pid<programs) (hl : lane<fp.block) :
    pid*bp.rowsPerProgram < 2147483648 ∧
    (pid+1)*bp.rowsPerProgram < 2147483648 ∧
    (pid*bp.rowsPerProgram)*bp.columns+lane < 2147483648 := by
  have he := Nat.mul_le_mul_right bp.rowsPerProgram (show pid+1≤programs by omega)
  have hs := Nat.mul_le_mul_right bp.rowsPerProgram (Nat.le_of_lt hp)
  have positive : 1≤fp.columns := by have := h.columns; omega
  have scale : programs*bp.rowsPerProgram≤programs*bp.rowsPerProgram*fp.columns := by
    simpa using Nat.mul_le_mul_left (programs*bp.rowsPerProgram) positive
  have bounded : programs*bp.rowsPerProgram < 2147483648 :=
    lt_of_le_of_lt (scale.trans (Nat.le_add_right _ _)) h.sourceIndexBounds.2.1
  refine ⟨lt_of_le_of_lt hs bounded,lt_of_le_of_lt he bounded,?_⟩
  rw [h.sameColumns]
  exact lt_of_le_of_lt (Nat.add_le_add (Nat.mul_le_mul_right fp.columns hs) (Nat.le_of_lt hl))
    h.sourceIndexBounds.2.1

/-- Every partial-gradient store offset includes the actual program id and
the full padded lane range; the store mask still selects lane<N. -/
theorem partial_index (fp : ForwardParameters) (bp : BackwardParameters) (rows programs pid lane : Nat)
    (h : Supported fp bp rows programs) (hp : pid<programs) (hl : lane<fp.block) :
    pid*bp.columns+lane < 2147483648 := by
  rw [h.sameColumns]
  exact lt_of_le_of_lt (Nat.add_le_add (Nat.mul_le_mul_right fp.columns (Nat.le_of_lt hp))
    (Nat.le_of_lt hl)) h.sourceIndexBounds.2.2

/-- Produced Mean/Rstd are contiguous. The absent RMS Mean stride is zero. -/
theorem saved_statistic_indices (fp : ForwardParameters) (bp : BackwardParameters)
    (rows programs row : Nat) (h : Supported fp bp rows programs) (hr : row<rows) :
    row*bp.meanStride < 2147483648 ∧ row*bp.rstdStride < 2147483648 := by
  have positive : 1≤fp.columns := by have := h.columns; omega
  have scale : rows≤rows*fp.columns := by simpa using Nat.mul_le_mul_left rows positive
  have bound : row < 2147483648 := lt_of_lt_of_le hr (scale.trans (Nat.le_of_lt h.rowsBound))
  rw [h.savedStrides.1,h.savedStrides.2]
  cases bp.rms <;> simp [bound]

/-- Every actual logical row belongs to exactly one launched backward program. -/
theorem row_ownership (fp : ForwardParameters) (bp : BackwardParameters) (rows programs row : Nat)
    (h : Supported fp bp rows programs) (hr : row<rows) :
    ∃! pid, pid<programs ∧ pid*bp.rowsPerProgram≤row ∧
      row<min ((pid+1)*bp.rowsPerProgram) bp.rows := by
  simpa only [h.width,h.sameRows,Launch.owns] using
    Launch.exactly_one_owner rows programs row h.programsPositive hr

theorem partition_sum (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (h : Supported fp bp rows programs) (f : Nat → ℝ) :
    (∑ pid ∈ Finset.range programs, ∑ row ∈ Finset.Ico (pid*bp.rowsPerProgram)
      (min ((pid+1)*bp.rowsPerProgram) bp.rows), f row) = ∑ row ∈ Finset.range rows, f row := by
  rw [h.width,h.sameRows]
  by_cases hz : rows=0
  · subst rows
    simp [Launch.rowsPerProgram]
  · exact Launch.group_sum rows programs h.programsPositive (by omega) f

#axiomsClean row_ownership
#axiomsClean partition_sum
#axiomsClean forward_index
#axiomsClean backward_program_indices
#axiomsClean partial_index
#axiomsClean saved_statistic_indices
end
end VeriTile.Bench.Repairs.LayerNormOps.Scope
