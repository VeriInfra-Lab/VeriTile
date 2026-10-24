import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulForward

namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {metadata : MetadataAPI api neg inputs Metadata}
  {service : Runtime.Service platform api neg inputs Metadata} {identity : Nat}
  {x w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem logicalBias_transport (tr : Transition api neg inputs before after)
    (bias : Option Storage.Tensor) (m : Storage.World (Nat → ℝ)) (col : Nat)
    (present : ∀ b,b∈bias.toList → before.live b)
    (valid : ∀ b,bias=some b → col<b.view.shape.prod) :
    logicalBias (tr.shadow m) bias col=logicalBias m bias col := by
  cases bias with
  | none => rfl
  | some b => exact tr.realValues m b (present b (by simp)) col (valid b rfl)

theorem forward_logical_real (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (xPresent : initial.live x) (weightPresent : initial.live w)
    (biasPresent : ∀ b,b∈bias.toList → initial.live b)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin x.view.shape.prod) :
    Storage.realLogical (r.realFinal m epsReal) r.body.output.tensor i.val=
      RealProof.outputSpec r.shape.columns (logicalX m x r.shape.columns (i.val/r.shape.columns))
        (logicalW m w) (logicalBias m bias) epsReal (i.val%r.shape.columns) := by
  have hret := output_storage r
  have hy := (route_counts r.body.preparation (shape_count r.shape)).2.1
  have hiy : i.val<r.body.preparation.data.y.count := by rw [hy,←shape_count r.shape]; exact i.isLt
  have live := (lib.viewed r.body.preparation.data.y x.view.shape r.body.sSave r.body.output r.body.viewed).present
  have hv := (lib.leave r.sShape.owner).transition r.body.final |>.realValues
    (r.body.realFinal (r.preTransition.shadow m) epsReal) r.body.output.tensor live i.val
    (by rw [hret.2.1]; exact i.isLt)
  change Storage.realLogical (r.realFinal m epsReal) r.body.output.tensor i.val=_ at hv
  rw [hv]
  have bodyResult := body_logical_real r.body (r.preTransition.live w weightPresent)
    (fun b hb => r.preTransition.live b (biasPresent b hb)) (r.preTransition.shadow m) epsReal ⟨i.val,hiy⟩
  simp only at bodyResult
  rw [bodyResult]
  have hc : i.val%r.shape.columns<r.shape.columns := Nat.mod_lt _ r.shape.positive
  have hr : i.val/r.shape.columns<r.shape.rows := by
    apply (Nat.div_lt_iff_lt_mul r.shape.positive).mpr
    rw [←shape_count r.shape]
    exact i.isLt
  apply RealProof.output_congr _ _ _ _ _ _ _ _ _ hc
  · intro col hcol
    have hi : (i.val/r.shape.columns)*r.shape.columns+col<x.view.shape.prod := by
      have hs := shape_count r.shape
      have hm := Nat.mul_le_mul_right r.shape.columns (show i.val/r.shape.columns+1≤r.shape.rows by omega)
      nlinarith
    exact r.preTransition.realValues m x xPresent _ hi
  · exact r.preTransition.realValues m w weightPresent _ (lt_of_lt_of_le hc r.checks.weight)
  · exact logicalBias_transport r.preTransition bias m _ biasPresent
      (fun b hb => lt_of_lt_of_le hc (r.checks.bias b hb))

#axiomsClean forward_logical_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
