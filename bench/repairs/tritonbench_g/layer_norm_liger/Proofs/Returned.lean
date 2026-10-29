import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridHost
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostResults
namespace VeriTile.Bench.Repairs.LayerNormLiger.Returned
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
open Logical
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- Allocator separation only; no numerical or output equation is assumed. -/
structure Fresh (p : Backward.Parameters) : Prop where
  dw : ∀ r, Reads p r → r≠p.dw
  db : ∀ r, Reads p r → r≠p.db
  dx : ∀ r, Reads p r → r≠p.dx

def before (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ) (pid : Nat) :=
  Grid.launch s (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s pid).mem pid

theorem inputs (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ) (pid : Nat)
    (fresh : Fresh p) (r : RegionName) (hr : Reads p r) :
    (before rest p s pid).mem r=s.mem r :=
  Grid.backward_inputs _ _ p s pid r (fresh.dw r hr) (fresh.db r hr) (fresh.dx r hr)

theorem groups (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ) (pid i : Nat)
    (fresh : Fresh p) :
    groupDW p (before rest p s pid) i=groupDW p (atPid s pid) i ∧
    groupDB p (before rest p s pid) i=groupDB p (atPid s pid) i ∧
    (∀ row, rowDX p (before rest p s pid) row i=rowDX p s row i) := by
  have hx := inputs rest p s pid fresh p.x (by simp [Reads])
  have hw := inputs rest p s pid fresh p.weight (by simp [Reads])
  have hy := inputs rest p s pid fresh p.dy (by simp [Reads])
  have hm := inputs rest p s pid fresh p.mean (by simp [Reads])
  have hr := inputs rest p s pid fresh p.rstd (by simp [Reads])
  unfold groupDW groupDB rowDX
  unfold bx bw bdy
  simp only [saved,Backward.mean,Backward.rstd,hx,hw,hy,hm,hr]
  simp [before,Grid.launch,atPid,Backward.start,Backward.stop]

theorem dw_cell (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs pid : Nat) (hp : pid<programs) (fresh : Fresh p)
    (hwd : p.dw≠p.db) (hwx : p.dw≠p.dx) (hs : p.columns≤p.dwStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs).mem
      p.dw (pid*p.dwStride+i.1.val)=.mk .fp32 (groupDW p (atPid s pid) i.1.val) := by
  rw [Grid.backward_dw _ _ p s programs pid hp hwd hwx hs i hi]
  have h := Logical.reference_dw rest p (before rest p s pid) fresh.dx hwd i hi
  unfold Backward.reference at h
  rw [Backward.finish_dw _ _ _ _ hwd i hi] at h
  rw [(groups rest p s pid i.1.val fresh).1] at h
  exact h

theorem db_cell (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs pid : Nat) (hp : pid<programs) (fresh : Fresh p)
    (hbw : p.db≠p.dw) (hbx : p.db≠p.dx) (hs : p.columns≤p.dbStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs).mem
      p.db (pid*p.dbStride+i.1.val)=.mk .fp32 (groupDB p (atPid s pid) i.1.val) := by
  rw [Grid.backward_db _ _ p s programs pid hp hbw hbx hs i hi]
  have h := Logical.reference_db rest p (before rest p s pid) fresh.dx i hi
  unfold Backward.reference at h
  rw [Backward.finish_db _ _ _ _ i hi] at h
  rw [(groups rest p s pid i.1.val fresh).2.1] at h
  exact h

theorem dx_cell (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs : Nat) (hp : 0<programs) (fresh : Fresh p)
    (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db) (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs)
    (row : Nat) (hr : row<p.rows) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs).mem
      p.dx (row*p.dxStride+i.1.val)=.mk .fp32 (rowDX p s row i.1.val) := by
  obtain ⟨pid,⟨hpid,hlo,hhi⟩,_⟩ := Logical.dx_coverage p s programs row hp hr hq
  have hlo' : Backward.start p (before rest p s pid)≤row := by
    simpa [Backward.start,before,Grid.launch,atPid] using hlo
  have hhi' : row<Backward.stop p (before rest p s pid) := by
    simpa [Backward.stop,before,Grid.launch,atPid] using hhi
  rw [Grid.backward_dx _ _ p s programs pid row hpid
    (by simpa [Backward.stop,atPid] using hhi) hxd hxb hs i hi]
  have h := Logical.reference_dx rest p (before rest p s pid) fresh.dx hxd hxb hb hs row hlo' hhi' i hi
  unfold Backward.reference at h
  rw [Backward.finish_frame _ _ _ _ _ _ (Or.inl hxd) (Or.inl hxb)] at h
  rw [(groups rest p s pid i.1.val fresh).2.2 row] at h
  exact h

def scalar : Host.Scalar ℝ := ⟨0,(·+·),id⟩

theorem tree_range (n : Nat) (p : ReductionPlan.Plan n) (f : Nat → ℝ) :
    p.tree.eval 0 (·+·) (fun i => f i.val)=∑ i ∈ Finset.range n, f i := by
  rw [p.real_sum,← Fin.sum_univ_def,Fin.sum_univ_eq_sum_range]

/-- The final Torch tensor reads the stored partial grid. The last step uses
the independently proved partition of logical rows, after memory readback. -/
theorem dw_return (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs : Nat) (hp : 0<programs) (hr : 0<p.rows) (fresh : Fresh p)
    (hwd : p.dw≠p.db) (hwx : p.dw≠p.dx) (hs : p.columns≤p.dwStride) (hb : p.columns≤p.block)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat) (hout : dw≠db)
    (i : Fin p.columns) :
    (Host.finish scalar p programs pw pb dw db shape
      (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs)).state.mem dw i.val=
      .mk .fp32 (Logical.dw p.rows (fun row => bx p (saved p s) row)
        (fun row => bdy p (saved p s) row) (fun row => Backward.mean p (saved p s) row)
        (fun row => Backward.rstd p (saved p s) row) i.val) := by
  rw [Host.finish_dw _ _ _ _ _ _ _ _ _ hout]
  let j : TileIndex [p.block] := (⟨i.val,Nat.lt_of_lt_of_le i.isLt hb⟩,PUnit.unit)
  have hcol : Host.column (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs)
      (Host.partialDW p programs) i.val=(fun pid => groupDW p (atPid s pid.val) i.val) := by
    funext pid
    dsimp [Host.column,Host.partialDW]
    rw [dw_cell rest p s programs pid.val pid.isLt fresh hwd hwx hs j i.isLt]
    simp [j]
  unfold Host.sumValue
  rw [hcol]
  change Cell.mk .fp32 (pw.tree.eval 0 (·+·) (fun pid => groupDW p (atPid s pid.val) i.val))=_
  rw [tree_range programs pw (fun pid => groupDW p (atPid s pid) i.val),
    Logical.host_dw p s programs hp hr hq i.val]

theorem db_return (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs : Nat) (hp : 0<programs) (hr : 0<p.rows) (fresh : Fresh p)
    (hbw : p.db≠p.dw) (hbx : p.db≠p.dx) (hs : p.columns≤p.dbStride) (hb : p.columns≤p.block)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat) (hout : p.db≠dw)
    (i : Fin p.columns) :
    (Host.finish scalar p programs pw pb dw db shape
      (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs)).state.mem db i.val=
      .mk .fp32 (Logical.db p.rows (fun row => bdy p (saved p s) row) i.val) := by
  rw [Host.finish_db _ _ _ _ _ _ _ _ _ hout]
  let j : TileIndex [p.block] := (⟨i.val,Nat.lt_of_lt_of_le i.isLt hb⟩,PUnit.unit)
  have hcol : Host.column (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs)
      (Host.partialDB p programs) i.val=(fun pid => groupDB p (atPid s pid.val) i.val) := by
    funext pid
    dsimp [Host.column,Host.partialDB]
    rw [db_cell rest p s programs pid.val pid.isLt fresh hbw hbx hs j i.isLt]
    simp [j]
  unfold Host.sumValue
  rw [hcol]
  change Cell.mk .fp32 (pb.tree.eval 0 (·+·) (fun pid => groupDB p (atPid s pid.val) i.val))=_
  rw [tree_range programs pb (fun pid => groupDB p (atPid s pid) i.val),
    Logical.host_db p s programs hp hr hq i.val]

#axiomsClean inputs
#axiomsClean groups
#axiomsClean dw_cell
#axiomsClean db_cell
#axiomsClean dx_cell
#axiomsClean tree_range
#axiomsClean dw_return
#axiomsClean db_return
end
end VeriTile.Bench.Repairs.LayerNormLiger.Returned
