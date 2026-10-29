import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ForwardCall
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RealCongruence

/-! Independent Real result for the same native body route. Arbitrary Real
caller storage and epsilon are chosen after fixing all native objects, API
outcomes, allocations, strides and launch schedules. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform} {block warps : Nat}
  {lib : APIHost api neg inputs Metadata} {service : Runtime.Service platform api neg inputs Metadata}
  {identity : Nat} {x w : Storage.Tensor} {bias : Option Storage.Tensor} {shape : ShapeInfo x}
  {checks : Checks x w bias shape block warps} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

def BodyRoute.preReal (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) := r.dispatch.preTransition.shadow (r.preparation.transition.shadow m)

def BodyRoute.realFinal (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) :=
  r.tail.shadow (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal)

def logicalX (m : Storage.World (Nat → ℝ)) (x : Storage.Tensor) (N row col : Nat) : ℝ :=
  Storage.realLogical m x (row*N+col)
def logicalW (m : Storage.World (Nat → ℝ)) (w : Storage.Tensor) (col : Nat) : ℝ :=
  Storage.realLogical m w col
def logicalBias (m : Storage.World (Nat → ℝ)) (bias : Option Storage.Tensor) (col : Nat) : ℝ :=
  match bias with | none => 0 | some b => Storage.realLogical m b col

theorem source_x_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (row col : Nat) (hr : row<shape.rows) (hc : col<shape.columns) :
    Execution.rows r.preparation.data.x shape.columns (r.preReal m) Memory.realAPI row col=
      logicalX m x shape.columns row col := by
  have hx := (route_counts r.preparation (shape_count shape)).1
  have hi : row*shape.columns+col<r.preparation.data.x.count :=
    (Execution.row_offset_safe _ _ _ _ _ hx hr hc).1
  have hio : row*shape.columns+col<x.view.shape.prod := by rw [shape_count shape]; rw [hx] at hi; exact hi
  have live := (route_present r.preparation).1 r.preparation.data.x (by simp [reads])
  have ht := r.dispatch.preTransition.realValues (r.preparation.transition.shadow m) _ live _
    (by rw [←r.preparation.data.x.countEq]; exact hi)
  have hp := prepare_x_real r.preparation m _ hio
  rw [Storage.dense_real _ _ _ hi] at ht
  exact ht.trans hp

theorem source_w_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (present : initial.live w) (m : Storage.World (Nat → ℝ)) (col : Nat) (hc : col<shape.columns) :
    Execution.vector r.preparation.data.w (r.preReal m) Memory.realAPI col=logicalW m w col := by
  have hw := (route_counts r.preparation (shape_count shape)).2.2.2.2
  have hi : col<r.preparation.data.w.count := by rw [hw]; exact lt_of_lt_of_le hc checks.weight
  have hio : col<w.view.shape.prod := lt_of_lt_of_le hc checks.weight
  have live := (route_present r.preparation).1 r.preparation.data.w (by simp [reads])
  have ht := r.dispatch.preTransition.realValues (r.preparation.transition.shadow m) _ live col
    (by rw [←r.preparation.data.w.countEq]; exact hi)
  rw [Storage.dense_real _ _ _ hi] at ht
  exact ht.trans (prepare_w_real r.preparation present m col hio)

theorem source_bias_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (present : ∀ b,b∈bias.toList → initial.live b) (m : Storage.World (Nat → ℝ))
    (col : Nat) (hc : col<shape.columns) :
    Runtime.biasValues RealProof.ops Memory.realAPI r.preparation.data (r.preReal m) col=logicalBias m bias col := by
  cases hbr : r.preparation.br with
  | none => simp [Runtime.biasValues,Route.data,hbr,OptionalRoute.output,logicalBias,RealProof.ops]
  | @some b br =>
    have hb : r.preparation.data.bias=some br.output := by simp [Route.data,hbr,OptionalRoute.output]
    have hio : col<b.view.shape.prod := lt_of_lt_of_le hc (checks.bias b rfl)
    have hi : col<br.output.count := by rw [(materialize_metadata br).2.1]; exact hio
    have live := (route_present r.preparation).1 br.output (by simp [reads,hb])
    have ht := r.dispatch.preTransition.realValues (r.preparation.transition.shadow m) _ live col
      (by rw [←br.output.countEq]; exact hi)
    rw [Storage.dense_real _ _ _ hi] at ht
    have hp := prepare_bias_real r.preparation b br.output rfl hb (present b (by simp)) m col hio
    simpa only [Runtime.biasValues,hb,logicalBias] using ht.trans hp

theorem body_real_readback (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin r.preparation.data.y.count) :
    Storage.realLogical (r.realFinal m epsReal) r.output.tensor i.val=
      RealProof.outputSpec shape.columns
        (Execution.rows r.preparation.data.x shape.columns (r.preReal m) Memory.realAPI (i.val/shape.columns))
        (Execution.vector r.preparation.data.w (r.preReal m) Memory.realAPI)
        (Runtime.biasValues RealProof.ops Memory.realAPI r.preparation.data (r.preReal m)) epsReal (i.val%shape.columns) := by
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  have hy := body_output_live r
  have hi : i.val<r.preparation.data.y.tensor.view.shape.prod := by rw [←r.preparation.data.y.countEq]; exact i.isLt
  have hsave := (lib.save r.context).transition r.dispatch.final |>.realValues
    (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal) _ hy i.val hi
  have hview := hv.realValues
    (((lib.save r.context).transition r.dispatch.final).shadow
      (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal)) i.val i.isLt
  change Storage.realLogical (r.realFinal m epsReal) r.output.tensor i.val=_ at hview
  rw [hview,hsave,Storage.dense_real _ _ _ i.isLt]
  have ho := route_output_owners r.preparation (2 : Fin 3)
  change r.preparation.data.y.owner=r.preparation.data.x.owner at ho
  rw [ho,Runtime.route_real_y_readback r.dispatch RealProof.ops (r.preparation.transition.shadow m) epsReal i]
  exact RealProof.forward_real _ _ _ _ _ _

theorem body_logical_real (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (weightPresent : initial.live w) (biasPresent : ∀ b,b∈bias.toList → initial.live b)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin r.preparation.data.y.count) :
    Storage.realLogical (r.realFinal m epsReal) r.output.tensor i.val=
      RealProof.outputSpec shape.columns (logicalX m x shape.columns (i.val/shape.columns))
        (logicalW m w) (logicalBias m bias) epsReal (i.val%shape.columns) := by
  rw [body_real_readback r m epsReal i]
  have hc : i.val%shape.columns<shape.columns := Nat.mod_lt _ shape.positive
  have hy := (route_counts r.preparation (shape_count shape)).2.1
  have hr : i.val/shape.columns<shape.rows := by
    apply (Nat.div_lt_iff_lt_mul shape.positive).mpr
    rw [←hy]
    exact i.isLt
  apply RealProof.output_congr _ _ _ _ _ _ _ _ _ hc
  · exact fun col hc => source_x_real r m _ col hr hc
  · exact source_w_real r weightPresent m _ hc
  · exact source_bias_real r biasPresent m _ hc

#axiomsClean source_x_real
#axiomsClean source_w_real
#axiomsClean source_bias_real
#axiomsClean body_real_readback
#axiomsClean body_logical_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
