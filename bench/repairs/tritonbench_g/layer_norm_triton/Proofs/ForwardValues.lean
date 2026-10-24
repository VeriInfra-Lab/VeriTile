import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardResult
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RealCongruence

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}
  {service : ForwardService platform api neg inputs Metadata} {p : Preparation.Data}
  {layout : Prepared p M N inputs} {settings : Compiler.settings N=.ok (B,Geometry.warps B)}
  {eps : Compiler.Shared.HostScalar} {initial : State Store Metadata}

def LaunchRoute.nativeValue (r : LaunchRoute service p layout settings eps initial) (i : Nat) : UInt32 :=
  match r with
  | .skipped _ => 0
  | .ran _ r => forwardValue (platform.scalar r.code.object.primitives) r.code.plan
      (Execution.E.rows p.x N r.s2.memory api (i/N)) (Execution.E.vector p.w r.s2.memory api)
      (Execution.E.vector (p.bias.getD p.w) r.s2.memory api) (platform.epsilon r.code.object eps) (i%N)

def LaunchRoute.preReal (r : LaunchRoute service p layout settings eps initial) (m : Storage.World (Nat → ℝ)) :=
  match r with | .skipped _ => m | .ran _ r => r.preTransition.shadow m

theorem launch_native (r : LaunchRoute service p layout settings eps initial) (i : Fin p.y.count) :
    Storage.logical api neg r.final.memory p.y.tensor i.val=r.nativeValue i.val := by
  cases r with
  | skipped inactive =>
    have hc := layout.yCount
    have hi := i.isLt
    simp only [not_and_or,ne_eq,not_not] at inactive
    rcases inactive with h|h <;> simp [h] at hc <;> omega
  | ran active r =>
    rw [dense_logical api neg _ _ i.val i.isLt]
    have ho := (layout.nonemptyLayout (by omega)).outputsOwned 2
    change p.y.owner=p.x.owner at ho
    rw [ho]
    exact Runtime.route_readback r 2 i

theorem launch_real (r : LaunchRoute service p layout settings eps initial) (m : Storage.World (Nat → ℝ))
    (epsilon : ℝ) (i : Fin p.y.count) :
    Storage.realLogical (r.realFinal m epsilon) p.y.tensor i.val=
      RealProof.Spec.output N (Execution.E.rows p.x N (r.preReal m) Memory.realAPI (i.val/N))
        (Execution.E.vector p.w (r.preReal m) Memory.realAPI)
        (Execution.E.vector (p.bias.getD p.w) (r.preReal m) Memory.realAPI) epsilon (i.val%N) := by
  cases r with
  | skipped inactive =>
    have hc := layout.yCount
    have hi := i.isLt
    simp only [not_and_or,ne_eq,not_not] at inactive
    rcases inactive with h|h <;> simp [h] at hc <;> omega
  | ran active r =>
    rw [dense_real _ _ i.val i.isLt]
    have ho := (layout.nonemptyLayout (by omega)).outputsOwned 2
    change p.y.owner=p.x.owner at ho
    rw [ho]
    exact Execution.forward_stored_real r.code.plan p (layout.nonemptyLayout (by omega)) epsilon
      (r.preTransition.shadow m) r.schedule (Runtime.route_coverage r) i

theorem launch_real_transport (r : LaunchRoute service p layout settings eps initial)
    (m : Storage.World (Nat → ℝ)) (t : Storage.Tensor) (present : initial.live t)
    (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.realLogical (r.preReal m) t i=Storage.realLogical m t i := by
  cases r with
  | skipped _ => rfl
  | ran _ r => exact r.preTransition.realValues m t present i hi

variable {lib : APIHost api neg inputs Metadata} {identity : Nat} {x w b : Storage.Tensor}
  {shape : Geometry.Shape x.view.shape w.view.shape b.view.shape} {B : Nat}
  {settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B)}

theorem body_native (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (i : Fin r.preparation.data.y.count) :
    Storage.logical api neg r.final.memory r.output.tensor i.val=r.launch.nativeValue i.val := by
  have hy : r.launch.final.live r.preparation.data.y.tensor := launch_live r.launch _ ((Layout.route_present r.preparation).2 2)
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  have hsave := (lib.save r.context).transition r.launch.final |>.values _ hy i.val
    (by rw [←r.preparation.data.y.countEq]; exact i.isLt)
  exact (hv.values i.val i.isLt).trans (hsave.trans (launch_native r.launch i))

theorem body_real_snapshot (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin r.preparation.data.y.count) :
    Storage.realLogical (r.realFinal m epsilon) r.output.tensor i.val=
      RealProof.Spec.output shape.columns
        (Execution.E.rows r.preparation.data.x shape.columns (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI (i.val/shape.columns))
        (Execution.E.vector r.preparation.data.w (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI)
        (Execution.E.vector (r.preparation.data.bias.getD r.preparation.data.w) (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI)
        epsilon (i.val%shape.columns) := by
  have hy : r.launch.final.live r.preparation.data.y.tensor := launch_live r.launch _ ((Layout.route_present r.preparation).2 2)
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  have hsave := (lib.save r.context).transition r.launch.final |>.realValues
    (r.launch.realFinal (r.preparation.transition.shadow m) epsilon) _ hy i.val
    (by rw [←r.preparation.data.y.countEq]; exact i.isLt)
  have hview := hv.realValues
    (((lib.save r.context).transition r.launch.final).shadow (r.launch.realFinal (r.preparation.transition.shadow m) epsilon)) i.val i.isLt
  exact hview.trans (hsave.trans (launch_real r.launch (r.preparation.transition.shadow m) epsilon i))

def logicalRow (m : Storage.World (Nat → ℝ)) (x : Storage.Tensor) (N row col : Nat) : ℝ :=
  Storage.realLogical m x (row*N+col)

theorem source_x_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (m : Storage.World (Nat → ℝ)) (row col : Nat) (hr : row<shape.rows) (hc : col<shape.columns) :
    Execution.E.rows r.preparation.data.x shape.columns (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI row col=
      logicalRow m x shape.columns row col := by
  have hx := (Layout.route_counts r.preparation (Geometry.shape_product shape).1).1
  have hi := (Geometry.source_row_offset_safe r.preparation.data.x shape.rows shape.columns row col hx hr hc).1
  have hio : row*shape.columns+col<x.view.shape.prod := by
    rw [(Geometry.shape_product shape).1]
    simpa only [hx] using hi
  have live := (Layout.route_present r.preparation).1 r.preparation.data.x (by simp [Layout.reads])
  have ht := launch_real_transport r.launch (r.preparation.transition.shadow m) _ live _
    (by rw [←r.preparation.data.x.countEq]; exact hi)
  rw [Storage.dense_real _ _ _ hi] at ht
  exact ht.trans (Preparation.prepare_x_real r.preparation m _ hio)

theorem source_w_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (present : initial.live w) (m : Storage.World (Nat → ℝ)) (col : Nat) (hc : col<shape.columns) :
    Execution.E.vector r.preparation.data.w (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI col=
      Storage.realLogical m w col := by
  have hw := (Layout.route_counts r.preparation (Geometry.shape_product shape).1).2.2.2.2
  have hio : col<w.view.shape.prod := by simpa only [(Geometry.shape_product shape).2.1] using hc
  have hi : col<r.preparation.data.w.count := by simpa only [hw] using hio
  have live := (Layout.route_present r.preparation).1 r.preparation.data.w (by simp [Layout.reads])
  have ht := launch_real_transport r.launch (r.preparation.transition.shadow m) _ live col
    (by rw [←r.preparation.data.w.countEq]; exact hi)
  rw [Storage.dense_real _ _ _ hi] at ht
  exact ht.trans (Preparation.prepare_w_real r.preparation present m col hio)

theorem source_bias_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (present : initial.live b) (m : Storage.World (Nat → ℝ)) (col : Nat) (hc : col<shape.columns) :
    Execution.E.vector (r.preparation.data.bias.getD r.preparation.data.w)
      (r.launch.preReal (r.preparation.transition.shadow m)) Memory.realAPI col=Storage.realLogical m b col := by
  cases hbr : r.preparation.br with
  | some br =>
    have hb : r.preparation.data.bias=some br.output := by simp [Preparation.Route.data,hbr,OptionalRoute.output]
    have hio : col<b.view.shape.prod := by simpa only [(Geometry.shape_product shape).2.2] using hc
    have hi : col<br.output.count := by rw [(materialize_metadata br).2.1]; exact hio
    have live := (Layout.route_present r.preparation).1 br.output (by simp [Layout.reads,hb])
    have ht := launch_real_transport r.launch (r.preparation.transition.shadow m) _ live col
      (by rw [←br.output.countEq]; exact hi)
    rw [Storage.dense_real _ _ _ hi] at ht
    simpa only [hb,Option.getD_some] using
      ht.trans (Preparation.prepare_bias_real r.preparation b br.output rfl hb present m col hio)

theorem body_original_real (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (weightPresent : initial.live w) (biasPresent : initial.live b)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) (i : Fin r.preparation.data.y.count) :
    Storage.realLogical (r.realFinal m epsilon) r.output.tensor i.val=
      RealProof.Spec.output shape.columns (logicalRow m x shape.columns (i.val/shape.columns))
        (Storage.realLogical m w) (Storage.realLogical m b) epsilon (i.val%shape.columns) := by
  rw [body_real_snapshot r m epsilon i]
  have hn : 0<shape.columns := by
    by_contra h
    have hz : shape.columns=0 := by omega
    have hc := r.layout.yCount
    have hc0 := hc.trans (Nat.mul_eq_zero.mpr (Or.inr hz))
    have hi := i.isLt
    omega
  have hc := Nat.mod_lt i.val hn
  have hr : i.val/shape.columns<shape.rows := by
    apply (Nat.div_lt_iff_lt_mul hn).mpr
    simpa only [←r.layout.yCount] using i.isLt
  apply VeriTile.Bench.Repairs.FastLayernorm.RealProof.output_congr _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => source_x_real r m _ col hr hc
  · exact source_w_real r weightPresent m _ hc
  · exact source_bias_real r biasPresent m _ hc

#axiomsClean body_native
#axiomsClean body_real_snapshot
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
