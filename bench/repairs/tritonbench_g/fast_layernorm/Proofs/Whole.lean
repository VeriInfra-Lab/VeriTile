import bench.repairs.tritonbench_g.fast_layernorm.Proofs.FullSaved
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.FullBackwardReal

/-! Whole forward/backward mathematical composition. The version receipt below
is the ordinary saved-tensor contract: it preserves the captured handles'
logical values, not a LayerNorm equation. Those values and statistics are
proved from the actual forward route in FullSaved. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Wrapper BackwardPreparation
noncomputable section
open scoped Classical

structure SavedVersion (api : Memory.API α Store Byte) (neg : α → α) (c : SavedContext)
    (captured current : State Store Metadata) where
  shadow : Storage.World (Nat → ℝ) → Storage.World (Nat → ℝ)
  live : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → current.live t.tensor
  values : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → ∀ i,i<t.count →
    Storage.logical api neg current.memory t.tensor i=Storage.logical api neg captured.memory t.tensor i
  realValues : ∀ m t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → ∀ i,i<t.count →
    Storage.realLogical (shadow m) t.tensor i=Storage.realLogical m t.tensor i

def backwardInputs (caller : List Storage.Tensor) (c : SavedContext) (dy : Storage.Tensor) : List Storage.Tensor :=
  caller++savedInputs c++[dy]

theorem saved_members (caller : List Storage.Tensor) (c : SavedContext) (dy : Storage.Tensor)
    (t : Dense) (ht : t∈[c.x,c.w,c.r,c.mu]++c.bias.toList) : t.tensor∈backwardInputs caller c dy := by
  simp only [backwardInputs,List.mem_append]
  exact Or.inl (Or.inr (List.mem_map.mpr ⟨t,ht,rfl⟩))

variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {caller : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg caller Metadata} {metadata : MetadataAPI api neg caller Metadata}
  {service : Runtime.Service platform api neg caller Metadata} {identity : Nat}
  {x w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
  {initial current : State Store Metadata}

variable (f : ForwardCall.SuccessfulCall lib metadata service identity x w bias eps initial)

theorem context_columns : f.body.context.columns=f.shape.columns := rfl
theorem context_rows : f.body.context.rows=f.shape.rows := rfl

def resumed (version : SavedVersion api neg f.body.context f.final current)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) := version.shadow (f.realFinal m epsReal)

theorem saved_x (version : SavedVersion api neg f.body.context f.final current)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ)
    (row col : Nat) (hr : row<f.shape.rows) (hc : col<f.shape.columns) :
    BackwardCall.inputX f.body.context (resumed f version m epsReal) row col=
      ForwardCall.logicalX m x f.shape.columns row col := by
  have hi : row*f.shape.columns+col<x.view.shape.prod := by
    have hs := shape_count f.shape
    have hm := Nat.mul_le_mul_right f.shape.columns (show row+1≤f.shape.rows by omega)
    nlinarith
  have hic : row*f.shape.columns+col<f.body.context.x.count := by
    have he := (ForwardCall.body_saved_layout f.body).inputShape
    change x.view.shape.prod=f.body.context.x.count at he
    omega
  exact (version.realValues _ _ (by simp) _ hic).trans (ForwardCall.saved_x_real f present m epsReal ⟨_,hi⟩)

theorem saved_w (version : SavedVersion api neg f.body.context f.final current)
    (present : initial.live w) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ)
    (col : Nat) (hc : col<f.shape.columns) :
    BackwardCall.inputW f.body.context (resumed f version m epsReal) col=ForwardCall.logicalW m w col := by
  have hi : col<w.view.shape.prod := lt_of_lt_of_le hc f.checks.weight
  have hic : col<f.body.context.w.count := lt_of_lt_of_le hc (ForwardCall.body_saved_layout f.body).wCount
  exact (version.realValues _ _ (by simp) col hic).trans (ForwardCall.saved_w_real f present m epsReal ⟨col,hi⟩)

theorem saved_stats (version : SavedVersion api neg f.body.context f.final current)
    (present : initial.live x) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ)
    (row : Nat) (hr : row<f.shape.rows) :
    BackwardCall.inputStats f.body.context (resumed f version m epsReal) row=
      ⟨RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row),
       RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row) epsReal⟩ := by
  have hm : row<f.body.context.mu.count := by rw [(ForwardCall.body_saved_layout f.body).muCount]; exact hr
  have hi : row<f.body.context.r.count := by rw [(ForwardCall.body_saved_layout f.body).rCount]; exact hr
  apply congrArg₂ Statistics.mk
  · exact (version.realValues _ _ (by simp) row hm).trans (ForwardCall.saved_mean_real f present m epsReal row hr)
  · exact (version.realValues _ _ (by simp) row hi).trans (ForwardCall.saved_inverse_real f present m epsReal row hr)

variable {dy : Storage.Tensor}
  {backLib : APIHost api neg (backwardInputs caller f.body.context dy) Metadata}
  {backService : BackwardRuntime.Service platform api neg (backwardInputs caller f.body.context dy) Metadata}

abbrev BackCall := BackwardCall.SuccessfulCall backLib backService identity f.body.context
  (ForwardCall.body_saved_layout f.body) dy current

def realFinal (version : SavedVersion api neg f.body.context f.final current)
    (b : BackCall f (current:=current) (backLib:=backLib) (backService:=backService)) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) :=
  b.realFinal (resumed f version m epsReal)

def incoming (version : SavedVersion api neg f.body.context f.final current)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) :=
  BackwardCall.inputDY f.body.context dy (resumed f version m epsReal)

theorem dx_real (version : SavedVersion api neg f.body.context f.final current)
    (b : BackCall f (current:=current) (backLib:=backLib) (backService:=backService))
    (xPresent : initial.live x) (wPresent : initial.live w) (dyPresent : current.live dy)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin b.body.preparation.data.dx.count) :
    Storage.realLogical (realFinal f version b m epsReal) b.body.output.1 i.val=
      RealProof.gradientSpec f.shape.columns
        (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns))
        (incoming f (dy:=dy) version m epsReal (i.val/f.shape.columns)) (ForwardCall.logicalW m w)
        (RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns)))
        (RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns (i.val/f.shape.columns)) epsReal)
        (i.val%f.shape.columns) := by
  have h := BackwardCall.dx_real b version.live dyPresent (resumed f version m epsReal) i
  change Storage.realLogical (realFinal f version b m epsReal) b.body.output.1 i.val=_ at h
  rw [h]
  simp only [context_columns,context_rows]
  have hc : i.val%f.shape.columns<f.shape.columns := Nat.mod_lt _ f.shape.positive
  have hr : i.val/f.shape.columns<f.shape.rows := by
    apply (Nat.div_lt_iff_lt_mul f.shape.positive).mpr
    have he := (BackwardLayout.route_layout b.body.preparation (ForwardCall.body_saved_layout f.body)).dxCount
    have hx := (ForwardCall.body_saved_layout f.body).xCount
    change f.body.context.x.count=f.shape.rows*f.shape.columns at hx
    have hi := i.isLt
    omega
  rw [saved_stats f version xPresent m epsReal _ hr]
  apply RealProof.gradient_congr _ _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => saved_x f version xPresent m epsReal _ col hr hc
  · intro col _; rfl
  · exact fun col hc => saved_w f version wPresent m epsReal col hc

theorem dw_real (version : SavedVersion api neg f.body.context f.final current)
    (b : BackCall f (current:=current) (backLib:=backLib) (backService:=backService))
    (xPresent : initial.live x) (dyPresent : current.live dy) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ)
    (i : Fin b.body.preparation.data.dw.count) :
    Storage.realLogical (realFinal f version b m epsReal) b.body.output.2.1 i.val=
      RealProof.weightGradientSpec f.shape.columns f.shape.rows (ForwardCall.logicalX m x f.shape.columns)
        (incoming f (dy:=dy) version m epsReal)
        (fun row => ⟨RealProof.meanSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row),
          RealProof.invSpec f.shape.columns (ForwardCall.logicalX m x f.shape.columns row) epsReal⟩) i.val := by
  have h := BackwardCall.dw_real b version.live (saved_members caller f.body.context dy) dyPresent
    (resumed f version m epsReal) i
  change Storage.realLogical (realFinal f version b m epsReal) b.body.output.2.1 i.val=_ at h
  rw [h]
  simp only [context_columns,context_rows]
  unfold RealProof.weightGradientSpec
  by_cases hc : i.val<f.shape.columns
  · simp only [hc,if_true]
    apply Finset.sum_congr rfl
    intro row hrow
    have hr : row<f.shape.rows := Finset.mem_range.mp hrow
    rw [saved_x f version xPresent m epsReal row i.val hr hc,saved_stats f version xPresent m epsReal row hr]
    rfl
  · simp only [hc,if_false]

theorem db_real (version : SavedVersion api neg f.body.context f.final current)
    (b : BackCall f (current:=current) (backLib:=backLib) (backService:=backService)) (db : Dense) (out : Storage.Tensor)
    (hdb : b.body.preparation.data.db=some db) (hout : b.body.output.2.2.1=some out)
    (dyPresent : current.live dy) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin db.count) :
    Storage.realLogical (realFinal f version b m epsReal) out i.val=
      RealProof.biasGradientSpec f.shape.columns f.shape.rows (incoming f (dy:=dy) version m epsReal) i.val :=
  BackwardCall.db_real b db out hdb hout dyPresent (resumed f version m epsReal) i

#axiomsClean saved_x
#axiomsClean saved_w
#axiomsClean saved_stats
#axiomsClean dx_real
#axiomsClean dw_real
#axiomsClean db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Whole
