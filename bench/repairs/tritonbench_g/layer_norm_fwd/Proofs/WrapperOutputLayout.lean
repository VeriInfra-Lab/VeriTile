import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLayout
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchState

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

theorem allocateOptional_property (api : TensorAPI world) (role : Role) (condition : Bool)
    (request : Allocation) (assertStride : Bool) (P : Tensor → Prop)
    (s : S) (output : Option Tensor)
    (h : ((allocateOptional api role condition request assertStride).run s).result=.ok output)
    (hp : ∀ current tensor, ((api.allocate role request).run current).2=.ok tensor →
      (assertStride=Bool.true → tensor.strides.getLast?=some 1) → P tensor) :
    OptionalFacts output P := by
  cases condition
  · have he : output=none := by simpa [allocateOptional,Pure.pure,Action.pure] using h.symm
    simp [OptionalFacts,he]
  · unfold allocateOptional at h
    simp only [Bool.true_eq,ite_true,Bind.bind] at h
    obtain ⟨tensor,ht,h,_⟩ := bind_success _ _ _ _ h
    cases ha : assertStride
    · simp only [ha,Bool.false_eq_true,ite_false] at h
      obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
      have he : output=some tensor := by simpa [Pure.pure,Action.pure] using h.symm
      intro out hout
      have ho : out=tensor := Option.some.inj (hout.symm.trans he)
      rw [ho]
      exact hp s tensor (call_success _ _ _ _ ht) (by simp [ha])
    · simp only [ha,ite_true] at h
      obtain ⟨u,hu,h,_⟩ := bind_success _ _ _ _ h
      have he : output=some tensor := by simpa [Pure.pure,Action.pure] using h.symm
      intro out hout
      have ho : out=tensor := Option.some.inj (hout.symm.trans he)
      rw [ho]
      exact hp s tensor (call_success _ _ _ _ ht)
        (fun _ => checkStride_success role tensor _ u hu)

structure OutputsDense (output : Outputs) : Prop where
  y : InnerDense output.y
  y1 : OptionalFacts output.y1 InnerDense
  residualOut : OptionalFacts output.residualOut InnerDense
  mean : OptionalFacts output.mean ContiguousLayout
  rstd : ContiguousLayout output.rstd
  seeds : OptionalFacts output.seeds ContiguousLayout
  mask : OptionalFacts output.mask ContiguousLayout
  y_injective : CoordinateInjective output.y
  y1_injective : OptionalFacts output.y1 CoordinateInjective
  residualOut_injective : OptionalFacts output.residualOut CoordinateInjective

/-- Matrix outputs use the source's successful inner-stride assertions;
ordinary empty/randint allocations supply dense logical offsets for the
scalar vectors and combined mask, including empty and singleton axes. -/
theorem allocateOutputs_dense (api : TensorAPI world) (hl : LayoutAPI api)
    (q : Request) (v : Validated) (input : Inputs) (s : S) (output : Outputs)
    (h : ((allocateOutputs api q v input).run s).result=.ok output) : OutputsDense output := by
  unfold allocateOutputs at h
  dsimp only [Bind.bind] at h
  obtain ⟨y,hyallocate,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨uy,hy,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨y1,hy1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨residual,hres,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨mean,hmean,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨rstd,hrstd,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨seeds,hseeds,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨mask,hmask,h,_⟩ := bind_success _ _ _ _ h
  have he : output=⟨y,y1,residual,mean,rstd,seeds,mask⟩ := by
    simpa [Pure.pure,Action.pure] using h.symm
  rw [he]
  refine ⟨innerDense_of_last y (checkStride_success .y y _ uy hy),?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · refine allocateOptional_property api .y1 _ _ Bool.true InnerDense _ y1 hy1 ?_
    intro current tensor _ hs
    exact innerDense_of_last tensor (hs rfl)
  · refine allocateOptional_property api .residualOut _ _ Bool.true InnerDense _ residual hres ?_
    intro current tensor _ hs
    exact innerDense_of_last tensor (hs rfl)
  · refine allocateOptional_property api .mean _ _ Bool.false ContiguousLayout _ mean hmean ?_
    intro current tensor hs _
    exact hl.allocate_contiguous .mean _ current tensor hs rfl
  · exact hl.allocate_contiguous .rstd _ _ rstd (call_success _ _ _ _ hrstd) rfl
  · refine allocateOptional_property api .seeds _ _ Bool.false ContiguousLayout _ seeds hseeds ?_
    intro current tensor hs _
    exact hl.allocate_contiguous .seeds _ current tensor hs rfl
  · refine allocateOptional_property api .mask _ _ Bool.false ContiguousLayout _ mask hmask ?_
    intro current tensor hs _
    exact hl.allocate_contiguous .mask _ current tensor hs rfl
  · exact hl.allocate_injective .y _ _ y (call_success _ _ _ _ hyallocate)
  · refine allocateOptional_property api .y1 _ _ Bool.true CoordinateInjective _ y1 hy1 ?_
    intro current tensor hs _
    exact hl.allocate_injective .y1 _ current tensor hs
  · refine allocateOptional_property api .residualOut _ _ Bool.true CoordinateInjective _ residual hres ?_
    intro current tensor hs _
    exact hl.allocate_injective .residualOut _ current tensor hs

theorem prepare_dense (api : TensorAPI world) (hl : LayoutAPI api)
    (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) : InputsDense p.inputs ∧ OutputsDense p.outputs := by
  unfold prepare at h
  dsimp only [Bind.bind] at h
  obtain ⟨v,hv,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨input,hi,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨output,ho,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨block,_,h,_⟩ := bind_success _ _ _ _ h
  have he : p=⟨v,input,output,block⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [he]
  exact ⟨resolveInputs_dense api hl q v (validate_facts q s v hv) _ input hi,
    allocateOutputs_dense api hl q v input _ output ho⟩

theorem route_dense (tensor : TensorAPI world) (device : DispatchAPI world)
    (hl : LayoutAPI tensor) (kernel : LaunchArguments → Action world Unit)
    (q : Request) (s : S) (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device kernel q s result p args) :
    ValidatedFacts q p.validated ∧ InputsDense p.inputs ∧ OutputsDense p.outputs :=
  ⟨prepare_validated tensor q s p route.prepared,prepare_dense tensor hl q s p route.prepared⟩

#axiomsClean allocateOptional_property
#axiomsClean allocateOutputs_dense
#axiomsClean prepare_dense
#axiomsClean route_dense
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
