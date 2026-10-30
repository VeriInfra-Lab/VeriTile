import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardResult
import bench.tritonbench_g.layer_norm_liger.Proofs.LogicalForward
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def atPid (s : State ℝ) (pid : Nat) : State ℝ :=
  { s with pids := fun ax => if ax=0 then pid else s.pids ax }

/-- Generic host sum over the stored DW partials is the full logical weight gradient. -/
theorem host_dw (p : Backward.Parameters) (s : State ℝ) (programs : Nat)
    (hp : 0<programs) (hr : 0<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) (i : Nat) :
    (∑ pid ∈ Finset.range programs, groupDW p (atPid s pid) i) =
      dw p.rows (fun row => bx p (saved p s) row) (fun row => bdy p (saved p s) row)
        (fun row => Backward.mean p (saved p s) row) (fun row => Backward.rstd p (saved p s) row) i := by
  simp only [groupDW, Backward.start, Backward.stop, atPid, if_true]
  change (∑ pid ∈ Finset.range programs,
    ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
      bdy p (saved p s) row i*xhat (bx p (saved p s) row)
        (Backward.mean p (saved p s) row) (Backward.rstd p (saved p s) row) i) = _
  rw [hq,Launch.group_sum p.rows programs hp hr]
  rfl

/-- Generic host sum over the stored DB partials is the full logical bias gradient. -/
theorem host_db (p : Backward.Parameters) (s : State ℝ) (programs : Nat)
    (hp : 0<programs) (hr : 0<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) (i : Nat) :
    (∑ pid ∈ Finset.range programs, groupDB p (atPid s pid) i) =
      db p.rows (fun row => bdy p (saved p s) row) i := by
  simp only [groupDB, Backward.start, Backward.stop, atPid, if_true]
  change (∑ pid ∈ Finset.range programs,
    ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
      bdy p (saved p s) row i) = _
  rw [hq,Launch.group_sum p.rows programs hp hr]
  rfl

/-- Each logical DX cell is produced by exactly one launched program. -/
theorem dx_coverage (p : Backward.Parameters) (s : State ℝ) (programs row : Nat)
    (hp : 0<programs) (hr : row<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) :
    ∃! pid, pid<programs ∧ Backward.start p (atPid s pid)≤row ∧ row<Backward.stop p (atPid s pid) := by
  simpa [Backward.start,Backward.stop,atPid,hq,Launch.owns] using
    Launch.exactly_one_owner p.rows programs row hp hr

/-- In a contiguous row-major tensor, every active FP32 access occupies a
complete word inside the allocation. Tensor base separation is external. -/
theorem complete_word_bound (rows columns row col : Nat) (hr : row<rows) (hc : col<columns) :
    4*(row*columns+col)+3 < 4*(rows*columns) := by
  have hmul : (row+1)*columns≤rows*columns := Nat.mul_le_mul_right _ (by omega)
  nlinarith

/-- Programs whose group starts beyond the last row do not load any row. -/
theorem empty_group (p : Backward.Parameters) (s : State ℝ) (h : p.rows≤Backward.start p s) :
    Backward.stop p s≤Backward.start p s := by
  exact Nat.le_trans (Nat.min_le_right _ _) h

#axiomsClean host_dw
#axiomsClean host_db
#axiomsClean dx_coverage
#axiomsClean complete_word_bound
#axiomsClean empty_group
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
