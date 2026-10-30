import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridCells
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostResults
namespace VeriTile.Bench.Repairs.LayerNormLiger.ForwardSaved
open VeriTile.Triton FP.Structural
open Logical
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

def Reads (p : Forward.Parameters) (r : RegionName) := r=p.x ∨ r=p.weight ∨ r=p.bias
def Fresh (p : Forward.Parameters) :=
  ∀ r, Reads p r → r≠p.mean ∧ r≠p.rstd ∧ r≠p.y

def before (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ) (row : Nat) :=
  Grid.launch s (Grid.reference (Forward.reference (algebra rest) sumPrimitive p) s row).mem row

theorem inputs (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ) (row : Nat)
    (fresh : Fresh p) :
    rowX p (before rest p s row)=rowX p (atPid s row) ∧
    weight p (before rest p s row)=weight p s ∧ bias p (before rest p s row)=bias p s := by
  have eqr : ∀ r, Reads p r → (before rest p s row).mem r=s.mem r := by
    intro r hr
    exact Grid.forward_inputs _ _ p s row r (fresh r hr).1 (fresh r hr).2.1 (fresh r hr).2.2
  have hx := eqr p.x (by simp [Reads])
  have hw := eqr p.weight (by simp [Reads])
  have hb := eqr p.bias (by simp [Reads])
  unfold rowX weight bias
  simp only [hx,hw,hb]
  simp [before,Grid.launch,atPid]

/-- Actual grid-produced saved statistics, for every logical row. -/
theorem statistics (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (rows row : Nat) (hr : row<rows) (fresh : Fresh p) (hb : p.columns≤p.block)
    (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y)
    (hms : p.meanStride=1) (hrs : p.rstdStride=1) :
    (Grid.reference (Forward.reference (algebra rest) sumPrimitive p) s rows).mem p.mean row=
      .mk .fp32 (mean p.columns (rowX p (atPid s row))) ∧
    (Grid.reference (Forward.reference (algebra rest) sumPrimitive p) s rows).mem p.rstd row=
      .mk .fp32 (rstd p.columns p.epsilon (rowX p (atPid s row))) := by
  have hm := Grid.forward_mean (algebra rest) sumPrimitive p s rows row hr hmy hmr (by omega)
  have hv := Grid.forward_rstd (algebra rest) sumPrimitive p s rows row hr hry hmr.symm (by omega)
  simp only [hms,hrs,Nat.mul_one] at hm hv
  rw [hm,hv]
  change Cell.mk .fp32 (Forward.meanValue (algebra rest) sumPrimitive p (before rest p s row))=_ ∧
    Cell.mk .fp32 (Forward.rstdValue (algebra rest) sumPrimitive p (before rest p s row))=_
  rw [Logical.forward_mean rest p _ hb,Logical.forward_rstd rest p _ hb,
    (inputs rest p s row fresh).1]
  exact ⟨rfl,rfl⟩

theorem output (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (rows row : Nat) (hr : row<rows) (fresh : Fresh p) (hb : p.columns≤p.block)
    (hym : p.y≠p.mean) (hyr : p.y≠p.rstd) (hs : p.columns≤p.yStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Forward.reference (algebra rest) sumPrimitive p) s rows).mem
      p.y (row*p.yStride+i.1.val)=.mk .fp32
        (y p.columns p.epsilon (rowX p (atPid s row)) (weight p s) (bias p s) i.1.val) := by
  rw [Grid.forward_y _ _ p s rows row hr hym hyr hs i hi]
  change Cell.mk .fp32 (Forward.yValue (algebra rest) sumPrimitive p (before rest p s row) i)=_
  rw [Logical.forward_y rest p _ hb i hi]
  obtain ⟨hx,hw,hb⟩ := inputs rest p s row fresh
  rw [hx,hw,hb]

/-- The backward loader consumes the forward-produced cells themselves.
Only region identity and normal contiguous statistics strides are required. -/
theorem saved_statistics (rest : Algebra ℝ) (fp : Forward.Parameters) (bp : Backward.Parameters)
    (s : State ℝ) (rows row : Nat) (hr : row<rows) (fresh : Fresh fp) (hb : fp.columns≤fp.block)
    (hmy : fp.mean≠fp.y) (hmr : fp.mean≠fp.rstd) (hry : fp.rstd≠fp.y)
    (hms : fp.meanStride=1) (hrs : fp.rstdStride=1) (hmean : bp.mean=fp.mean) (hrstd : bp.rstd=fp.rstd) :
    Backward.mean bp (saved bp (Grid.reference (Forward.reference (algebra rest) sumPrimitive fp) s rows)) row=
      mean fp.columns (rowX fp (atPid s row)) ∧
    Backward.rstd bp (saved bp (Grid.reference (Forward.reference (algebra rest) sumPrimitive fp) s rows)) row=
      rstd fp.columns fp.epsilon (rowX fp (atPid s row)) := by
  obtain ⟨hm,hv⟩ := statistics rest fp s rows row hr fresh hb hmy hmr hry hms hrs
  simp [Backward.mean,Backward.rstd,saved,hmean,hrstd,hm,hv]

#axiomsClean inputs
#axiomsClean statistics
#axiomsClean output
#axiomsClean saved_statistics
end
end VeriTile.Bench.Repairs.LayerNormLiger.ForwardSaved
