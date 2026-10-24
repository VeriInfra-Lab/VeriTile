import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SavedValues
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardCallReal
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Whole

namespace VeriTile.Bench.Repairs.LayerNormTriton.Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {caller : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {lib : APIHost api neg caller Metadata}
  {service : Host.ForwardService platform api neg caller Metadata}
  {identity : Nat} {x w bias : Storage.Tensor} {normalized : List Nat} {eps : Compiler.Shared.HostScalar}
  {initial current : State Store Metadata}

variable (f : Host.Call lib service identity x w bias normalized eps initial)
  {b : Dense} (saved : BackwardHost.Saved f.body.context b)

theorem context_columns : f.body.context.columns=f.shape.columns := rfl
theorem context_rows : f.body.context.rows=f.shape.rows := rfl

abbrev SavedVersion := @VeriTile.Bench.Repairs.FastLayernorm.Whole.SavedVersion

def resumed (version : SavedVersion api neg f.body.context f.final current)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) := version.shadow (f.realFinal m epsilon)

include saved in
theorem saved_x (version : SavedVersion api neg f.body.context f.final current) (present : initial.live x)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (row col : Nat) (hr : row<f.shape.rows) (hc : col<f.shape.columns) :
    BackwardHost.inputX f.body.context (resumed f version m epsilon) row col=Host.logicalRow m x f.shape.columns row col := by
  have hi : row*f.shape.columns+col<f.body.context.x.count := by
    rw [saved.xCount]
    exact BackwardHost.row_index (c:=f.body.context) row col hr hc
  exact (version.realValues _ _ (by simp) _ hi).trans (Host.saved_x_real f present m epsilon row col hr hc)

include saved in
theorem saved_w (version : SavedVersion api neg f.body.context f.final current) (present : initial.live w)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (col : Nat) (hc : col<f.shape.columns) :
    BackwardHost.inputW f.body.context (resumed f version m epsilon) col=Storage.realLogical m w col := by
  have hi : col<f.body.context.w.count := by rw [saved.wCount]; exact hc
  exact (version.realValues _ _ (by simp) _ hi).trans (Host.saved_w_real f present m epsilon col hc)

include saved in
theorem saved_stats (version : SavedVersion api neg f.body.context f.final current) (present : initial.live x)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (hn : 0<f.shape.columns) (row : Nat) (hr : row<f.shape.rows) :
    BackwardHost.inputStats f.body.context (resumed f version m epsilon) row=
      ⟨RealProof.Spec.mean f.shape.columns (Host.logicalRow m x f.shape.columns row),
       RealProof.Spec.inverse f.shape.columns (Host.logicalRow m x f.shape.columns row) epsilon⟩ := by
  have hm : row<f.body.context.mu.count := by rw [saved.muCount]; exact hr
  have hv : row<f.body.context.r.count := by rw [saved.rCount]; exact hr
  apply Eq.trans (congrArg₂ VeriTile.Bench.Repairs.FastLayernorm.Statistics.mk
    (version.realValues _ _ (by simp) row hm) (version.realValues _ _ (by simp) row hv))
  exact Host.saved_stats_real f present m epsilon hn row hr

variable {dy : Storage.Tensor} {backInputs : List Storage.Tensor}
  {backLib : APIHost api neg backInputs Metadata} {extra : BackwardHost.APIExtra api neg backInputs Metadata}
  {backService : BackwardHost.Service platform api neg backInputs Metadata}

abbrev BackCall := BackwardHost.Call backLib extra backService identity f.body.context b saved dy current

def realFinal (version : SavedVersion api neg f.body.context f.final current)
    (back : BackCall f saved (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService))
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) := back.realFinal (resumed f version m epsilon)

def incoming (version : SavedVersion api neg f.body.context f.final current)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) := BackwardHost.inputDY f.body.context dy (resumed f version m epsilon)

theorem dx_real (version : SavedVersion api neg f.body.context f.final current)
    (back : BackCall f saved (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService))
    (xp : initial.live x) (wp : initial.live w) (dyp : current.live dy)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin back.body.preparation.data.base.dx.count) :
    Storage.realLogical (realFinal f saved version back m epsilon) back.body.returns.dx.tensor i.val=
      RealProof.Spec.gradient f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns))
        (incoming f (dy:=dy) version m epsilon (i.val/f.shape.columns)) (Storage.realLogical m w)
        (RealProof.Spec.mean f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns)))
        (RealProof.Spec.inverse f.shape.columns (Host.logicalRow m x f.shape.columns (i.val/f.shape.columns)) epsilon)
        (i.val%f.shape.columns) := by
  have h := BackwardHost.returned_dx_real back version.live dyp (resumed f version m epsilon) i
  change Storage.realLogical (realFinal f saved version back m epsilon) back.body.returns.dx.tensor i.val=_ at h
  rw [h]
  have hi : i.val<f.shape.rows*f.shape.columns := lt_of_lt_of_eq i.isLt back.body.ready.base.dxCount
  have hn : 0<f.shape.columns := Nat.pos_of_mul_pos_left (lt_of_le_of_lt (Nat.zero_le _) hi)
  have hr := (Nat.div_lt_iff_lt_mul hn).mpr hi
  have hc := Nat.mod_lt i.val hn
  simp only [context_columns,context_rows]
  rw [saved_stats f saved version xp m epsilon hn _ hr]
  apply VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => saved_x f saved version xp m epsilon _ col hr hc
  · intro col _; rfl
  · exact fun col hc => saved_w f saved version wp m epsilon col hc

theorem dw_real (version : SavedVersion api neg f.body.context f.final current)
    (back : BackCall f saved (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService))
    (xp : initial.live x) (dyp : current.live dy)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin back.body.preparation.data.base.dw.count) :
    Storage.realLogical (realFinal f saved version back m epsilon) back.body.returns.wr.output i.val=
      RealProof.Spec.weightGradient f.shape.columns f.shape.rows (Host.logicalRow m x f.shape.columns)
        (incoming f (dy:=dy) version m epsilon)
        (fun row => ⟨RealProof.Spec.mean f.shape.columns (Host.logicalRow m x f.shape.columns row),
          RealProof.Spec.inverse f.shape.columns (Host.logicalRow m x f.shape.columns row) epsilon⟩) i.val := by
  have h := BackwardHost.returned_dw_real back version.live dyp (resumed f version m epsilon) i
  change Storage.realLogical (realFinal f saved version back m epsilon) back.body.returns.wr.output i.val=_ at h
  rw [h]
  have hc : i.val<f.shape.columns := by simpa only [back.body.ready.base.dwCount] using i.isLt
  have hn : 0<f.shape.columns := lt_of_le_of_lt (Nat.zero_le _) hc
  simp only [RealProof.Spec.weightGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec]
  change (if i.val<f.shape.columns then _ else 0)=(if i.val<f.shape.columns then _ else 0)
  simp only [hc,if_true]
  apply Finset.sum_congr rfl
  intro row hrow
  have hr := Finset.mem_range.mp hrow
  rw [saved_x f saved version xp m epsilon row i.val hr hc,saved_stats f saved version xp m epsilon hn row hr]
  rfl

theorem db_real (version : SavedVersion api neg f.body.context f.final current)
    (back : BackCall f saved (current:=current) (dy:=dy) (backLib:=backLib) (extra:=extra) (backService:=backService))
    (dyp : current.live dy) (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin back.body.preparation.data.base.db.count) :
    Storage.realLogical (realFinal f saved version back m epsilon) back.body.returns.br.output i.val=
      RealProof.Spec.biasGradient f.shape.columns f.shape.rows (incoming f (dy:=dy) version m epsilon) i.val :=
  BackwardHost.returned_db_real back dyp (resumed f version m epsilon) i

#axiomsClean dx_real
#axiomsClean dw_real
#axiomsClean db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Whole
