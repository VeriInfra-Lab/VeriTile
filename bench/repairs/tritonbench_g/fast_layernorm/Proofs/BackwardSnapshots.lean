import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardReal

/-! Launch snapshots are related to the original dY view and actual saved
handles. Preservation through the dX launch is proved from its stores. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

def BodyRoute.beforeDX (r : BodyRoute lib service c saved dy initial) : Transition api neg inputs initial r.dx.s2 :=
  r.preparation.transition.then r.dx.preTransition

theorem dx_input_real (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (t : Dense) (present : initial.live t.tensor) (i : Fin t.count) :
    r.dxInput m t.owner (t.address i.val)=Storage.realLogical m t.tensor i.val := by
  have h := r.beforeDX.realValues m t.tensor present i.val (by rw [←t.countEq]; exact i.isLt)
  rw [Storage.dense_real _ _ _ i.isLt] at h
  exact h

theorem dx_dy_real (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.dy.count) :
    r.dxInput m r.preparation.data.dy.owner (r.preparation.data.dy.address i.val)=Storage.realLogical m dy i.val := by
  have hm := Preparation.materialize_metadata r.preparation.dyr
  have h := r.dx.preTransition.realValues (r.preparation.transition.shadow m) r.preparation.data.dy.tensor
    (preparation_present r.preparation).1 i.val (by rw [←r.preparation.data.dy.countEq]; exact i.isLt)
  rw [Storage.dense_real _ _ _ i.isLt] at h
  exact h.trans (BackwardPreparation.dy_real r.preparation m i.val (by rw [←hm.2.1]; exact i.isLt))

theorem affine_input_real (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (t : Dense) (present : initial.live t.tensor) (member : t.tensor∈inputs) (owner : t.owner=c.x.owner)
    (i : Fin t.count) :
    r.affineInput m t.owner (t.address i.val)=Storage.realLogical m t.tensor i.val := by
  have live := r.dx.preTransition.live _ (r.preparation.transition.live _ present)
  have h := r.affine.preTransition.realValues (r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m))
    t.tensor live i.val (by rw [←t.countEq]; exact i.isLt)
  rw [Storage.dense_real _ _ _ i.isLt,Storage.dense_real _ _ _ i.isLt] at h
  have hprefix := BackwardRuntime.preserved_input c RealProof.ops r.dx.code.plan Memory.realAPI r.preparation.data
    (route_layout r.preparation saved) .dx r.dx.schedule (r.dxInput m) t member owner i
  exact h.trans (hprefix.trans (dx_input_real r m t present i))

theorem affine_dy_real (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (i : Fin r.preparation.data.dy.count) :
    r.affineInput m r.preparation.data.dy.owner (r.preparation.data.dy.address i.val)=Storage.realLogical m dy i.val := by
  let layout := route_layout r.preparation saved
  have live := r.dx.preTransition.live _ (preparation_present r.preparation).1
  have h := r.affine.preTransition.realValues (r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m))
    r.preparation.data.dy.tensor live i.val (by rw [←r.preparation.data.dy.countEq]; exact i.isLt)
  rw [Storage.dense_real _ _ _ i.isLt,Storage.dense_real _ _ _ i.isLt] at h
  have hprefix := trace_prefix_input Memory.realAPI
    (BackwardRuntime.trace c RealProof.ops r.dx.code.plan Memory.realAPI r.preparation.data layout .dx (r.dxInput m))
    (r.dxInput m) r.preparation.data.dy layout.dyOwner
    (fun k => layout.freshDy _ (by simp [BackwardRuntime.trace,dxTrace,backwardRegion,outputs]))
    _ (scheduled_subset _ r.dx.schedule) i
  exact h.trans (hprefix.trans (dx_dy_real r m i))

def inputX (c : SavedContext) (m : Storage.World (Nat → ℝ)) (row col : Nat) :=
  Storage.realLogical m c.x.tensor (row*c.columns+col)
def inputW (c : SavedContext) (m : Storage.World (Nat → ℝ)) (col : Nat) := Storage.realLogical m c.w.tensor col
def inputDY (c : SavedContext) (dy : Storage.Tensor) (m : Storage.World (Nat → ℝ)) (row col : Nat) :=
  Storage.realLogical m dy (row*c.columns+col)
def inputStats (c : SavedContext) (m : Storage.World (Nat → ℝ)) (row : Nat) : Statistics ℝ :=
  ⟨Storage.realLogical m c.mu.tensor row,Storage.realLogical m c.r.tensor row⟩

theorem dx_saved_stats (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (row : Nat) (hr : row<c.rows) :
    Execution.saved Memory.realAPI c.r c.mu (r.dxInput m) row=inputStats c m row := by
  have hm : row<c.mu.count := by rw [saved.muCount]; exact hr
  have hi : row<c.r.count := by rw [saved.rCount]; exact hr
  apply congrArg₂ Statistics.mk
  · exact dx_input_real r m c.mu (present _ (by simp)) ⟨row,hm⟩
  · exact dx_input_real r m c.r (present _ (by simp)) ⟨row,hi⟩

theorem affine_saved_stats (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ))
    (present : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → initial.live t.tensor)
    (members : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → t.tensor∈inputs)
    (row : Nat) (hr : row<c.rows) :
    Execution.saved Memory.realAPI c.r c.mu (r.affineInput m) row=inputStats c m row := by
  have hm : row<c.mu.count := by rw [saved.muCount]; exact hr
  have hi : row<c.r.count := by rw [saved.rCount]; exact hr
  apply congrArg₂ Statistics.mk
  · exact affine_input_real r m c.mu (present _ (by simp)) (members _ (by simp))
      (saved.owners _ (by simp)) ⟨row,hm⟩
  · exact affine_input_real r m c.r (present _ (by simp)) (members _ (by simp))
      (saved.owners _ (by simp)) ⟨row,hi⟩

#axiomsClean dx_input_real
#axiomsClean dx_dy_real
#axiomsClean affine_input_real
#axiomsClean affine_dy_real
#axiomsClean dx_saved_stats
#axiomsClean affine_saved_stats
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
