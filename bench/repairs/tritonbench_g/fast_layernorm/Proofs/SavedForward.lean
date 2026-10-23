import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulForward
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPreparation

/-! Saved-context shape facts are extracted from the actual forward route.
Saved mean/inverse values are then derived from its concrete stores. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper BackwardPreparation
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform} {block warps : Nat}
  {lib : APIHost api neg inputs Metadata} {service : Runtime.Service platform api neg inputs Metadata}
  {identity : Nat} {x w : Storage.Tensor} {bias : Option Storage.Tensor} {shape : ShapeInfo x}
  {checks : Checks x w bias shape block warps} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem body_saved_layout (r : BodyRoute lib service identity x w bias shape checks eps initial) :
    SavedLayout r.context := by
  let h := Layout.route_layout r.preparation shape.positive (shape_count shape) checks.weight checks.bias
  refine ⟨shape.positive,(materialize_metadata r.preparation.xr).1,h.xCount,h.rCount,h.muCount,h.wCount,h.biasCount,?_,?_,?_,?_,?_,?_,?_,checks.settings⟩
  · intro t ht
    change t∈[r.preparation.data.x,r.preparation.data.w,r.preparation.data.r,r.preparation.data.mu]++r.preparation.data.bias.toList at ht
    simp only [List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with (hx|hw|hr|hm)|hb
    · subst t; rfl
    · subst t; exact h.inputsOwned _ (by simp [reads])
    · subst t; exact h.outputsOwned (0 : Fin 3)
    · subst t; exact h.outputsOwned (1 : Fin 3)
    · exact h.inputsOwned _ (by simp [reads,hb])
  · exact (materialize_metadata r.preparation.xr).2.1.symm
  · exact (materialize_metadata r.preparation.wr).2.1.symm
  · intro b hb
    change r.preparation.data.bias=some b at hb
    cases hbr : r.preparation.br with
    | none => simp [Preparation.Route.data,hbr,OptionalRoute.output] at hb
    | @some original br =>
      have he : b=br.output := by simpa [Preparation.Route.data,hbr,OptionalRoute.output] using hb.symm
      subst b
      exact ⟨original.view.shape,rfl,(materialize_metadata br).2.1.symm⟩
  · cases hbr : r.preparation.br <;> simp [BodyRoute.context,savedContext,Preparation.Route.data,hbr,OptionalRoute.output]
  · let dense := r.preparation.wr.dense
    have hd := (lib.contiguousResult x.view.device w _ dense.output dense.contiguousOK).1.owner
    have hv := lib.viewed dense.output _ dense.final r.preparation.wr.output r.preparation.wr.viewOK
    change w.view.device=r.preparation.wr.output.owner
    rw [show r.preparation.wr.output.owner=dense.output.owner from congrArg (fun b : Storage.Backing => b.owner) hv.backing]
    exact hd.symm.trans dense.output.tensor.owner
  · intro b hb
    change r.preparation.data.bias=some b at hb
    cases hbr : r.preparation.br with
    | none => simp [Preparation.Route.data,hbr,OptionalRoute.output] at hb
    | @some original br =>
      have he : b=br.output := by simpa [Preparation.Route.data,hbr,OptionalRoute.output] using hb.symm
      subst b
      have hd := (lib.contiguousResult x.view.device original _ br.dense.output br.dense.contiguousOK).1.owner
      have hv := lib.viewed br.dense.output _ br.dense.final br.output br.viewOK
      change original.view.device=br.output.owner
      rw [show br.output.owner=br.dense.output.owner from congrArg (fun b : Storage.Backing => b.owner) hv.backing]
      exact hd.symm.trans br.dense.output.tensor.owner

theorem body_saved_live (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (t : Dense) (ht : t∈[r.context.x,r.context.w,r.context.r,r.context.mu]++r.context.bias.toList) :
    r.final.live t.tensor := by
  have present := route_present r.preparation
  apply r.tail.live
  apply r.dispatch.preTransition.live
  change t∈[r.preparation.data.x,r.preparation.data.w,r.preparation.data.r,r.preparation.data.mu]++r.preparation.data.bias.toList at ht
  simp only [List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with (hx|hw|hr|hm)|hb
  · subst t; exact present.1 _ (by simp [reads])
  · subst t; exact present.1 _ (by simp [reads])
  · subst t; exact present.2 (0 : Fin 3)
  · subst t; exact present.2 (1 : Fin 3)
  · exact present.1 _ (by simp [reads,hb])

theorem body_real_mean (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (row : Nat) (hr : row<shape.rows) :
    Storage.realLogical (r.realFinal m epsReal) r.context.mu.tensor row=
      RealProof.meanSpec shape.columns (logicalX m x shape.columns row) := by
  have hm : row<r.preparation.data.mu.count := by rw [(route_counts r.preparation (shape_count shape)).2.2.2.1]; exact hr
  have live := r.dispatch.preTransition.live _ ((route_present r.preparation).2 (1 : Fin 3))
  have hv := r.tail.realValues (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal)
    r.preparation.data.mu.tensor live row (by rw [←r.preparation.data.mu.countEq]; exact hm)
  change Storage.realLogical (r.realFinal m epsReal) r.context.mu.tensor row=_ at hv
  rw [hv,Storage.dense_real _ _ _ hm]
  have ho := route_output_owners r.preparation (1 : Fin 3)
  change r.preparation.data.mu.owner=r.preparation.data.x.owner at ho
  rw [ho]
  have read := Runtime.forward_mean_readback RealProof.ops r.dispatch.code.plan Memory.realAPI
    r.preparation.data (Layout.route_layout r.preparation shape.positive (shape_count shape) checks.weight checks.bias) epsReal r.dispatch.schedule (Runtime.route_coverage r.dispatch)
    (r.preReal m) ⟨row,hm⟩
  change r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal
    r.preparation.data.x.owner (r.preparation.data.mu.address row)=_ at read
  rw [read,RealProof.statistics_real]
  exact RealProof.mean_congr _ _ _ fun col hc => source_x_real r m row col hr hc

theorem body_real_inverse (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (row : Nat) (hr : row<shape.rows) :
    Storage.realLogical (r.realFinal m epsReal) r.context.r.tensor row=
      RealProof.invSpec shape.columns (logicalX m x shape.columns row) epsReal := by
  have hm : row<r.preparation.data.r.count := by rw [(route_counts r.preparation (shape_count shape)).2.2.1]; exact hr
  have live := r.dispatch.preTransition.live _ ((route_present r.preparation).2 (0 : Fin 3))
  have hv := r.tail.realValues (r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal)
    r.preparation.data.r.tensor live row (by rw [←r.preparation.data.r.countEq]; exact hm)
  change Storage.realLogical (r.realFinal m epsReal) r.context.r.tensor row=_ at hv
  rw [hv,Storage.dense_real _ _ _ hm]
  have ho := route_output_owners r.preparation (0 : Fin 3)
  change r.preparation.data.r.owner=r.preparation.data.x.owner at ho
  rw [ho]
  have read := Runtime.forward_inverse_readback RealProof.ops r.dispatch.code.plan Memory.realAPI
    r.preparation.data (Layout.route_layout r.preparation shape.positive (shape_count shape) checks.weight checks.bias) epsReal r.dispatch.schedule (Runtime.route_coverage r.dispatch)
    (r.preReal m) ⟨row,hm⟩
  change r.dispatch.realFinal RealProof.ops (r.preparation.transition.shadow m) epsReal
    r.preparation.data.x.owner (r.preparation.data.r.address row)=_ at read
  rw [read,RealProof.statistics_real]
  exact RealProof.inv_congr _ _ _ (fun col hc => source_x_real r m row col hr hc) epsReal

#axiomsClean body_saved_layout
#axiomsClean body_saved_live
#axiomsClean body_real_mean
#axiomsClean body_real_inverse
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
