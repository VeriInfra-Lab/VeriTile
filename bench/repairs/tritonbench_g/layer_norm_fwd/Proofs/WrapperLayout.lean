import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperValidation
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperTensorMemory

/-! Per-operation layout contracts. An inactive axis may have any stride:
only offsets of actual coordinates are constrained. These are ordinary
resolve_neg / contiguous-allocation properties, not kernel output laws. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def InnerDense (tensor : Tensor) : Prop :=
  ∀ col, col<tensor.shape.getLastD 0 → col*(tensor.strides.getLastD 0).toNat=col

def rowMajorOffset : List Nat → List Nat → Nat
  | _::sizes,index::indices => index*sizes.prod+rowMajorOffset sizes indices
  | _,_ => 0

def ContiguousLayout (tensor : Tensor) : Prop :=
  ∀ index, Coordinate tensor index →
    coordinateOffset tensor.strides index=rowMajorOffset tensor.shape index

def CoordinateInjective (tensor : Tensor) : Prop :=
  ∀ a b, Coordinate tensor a → Coordinate tensor b →
    coordinateOffset tensor.strides a=coordinateOffset tensor.strides b → a=b

variable {S Key Value : Type} {world : World S Key Value}

structure LayoutAPI (api : TensorAPI world) : Prop where
  resolve_inner : ∀ role input s output, ((api.resolve role input).run s).2=.ok output →
    input.strides.getLast?=some 1 → InnerDense output
  resolve_contiguous : ∀ role input s output, ((api.resolve role input).run s).2=.ok output →
    input.contiguous=Bool.true → ContiguousLayout output
  transfer_inner : ∀ target role input s output, ((api.transfer target role input).run s).2=.ok output →
    InnerDense input → InnerDense output
  transfer_contiguous : ∀ target role input s output, ((api.transfer target role input).run s).2=.ok output →
    ContiguousLayout input → ContiguousLayout output
  allocate_contiguous : ∀ role request s output, ((api.allocate role request).run s).2=.ok output →
    request.like=none → ContiguousLayout output
  allocate_injective : ∀ role request s output, ((api.allocate role request).run s).2=.ok output →
    CoordinateInjective output

theorem innerDense_of_last (tensor : Tensor) (h : tensor.strides.getLast?=some 1) : InnerDense tensor := by
  intro col _
  simp [List.getLastD_eq_getLast?,h]

theorem matrix_active_offset (tensor : Tensor) (M N : Nat) (stride : Int)
    (hs : tensor.shape=[M,N]) (hl : tensor.strides.length=tensor.shape.length)
    (hf : tensor.strides[0]?=some stride) (hd : InnerDense tensor)
    (row col : Nat) (hc : col<N) :
    coordinateOffset tensor.strides [row,col]=row*stride.toNat+col := by
  have hn : tensor.strides.length=2 := by simpa [hs] using hl
  obtain ⟨s,t,he⟩ : ∃ s t, tensor.strides=[s,t] := by
    exact ⟨tensor.strides[0]'(by omega),tensor.strides[1]'(by omega),
      List.eq_getElem_of_length_eq_two tensor.strides hn⟩
  have hfirst : s=stride := by simpa [he] using hf
  have hinner := hd col (by simpa [hs] using hc)
  simpa [he,hfirst,coordinateOffset] using congrArg (fun x => row*stride.toNat+x) hinner

theorem vector_active_offset (tensor : Tensor) (N : Nat)
    (hs : tensor.shape=[N]) (hl : tensor.strides.length=tensor.shape.length)
    (hd : InnerDense tensor) (index : Nat) (hi : index<N) :
    coordinateOffset tensor.strides [index]=index := by
  have hn : tensor.strides.length=1 := by simpa [hs] using hl
  obtain ⟨s,he⟩ := List.length_eq_one_iff.mp hn
  have hinner := hd index (by simpa [hs] using hi)
  simpa [he,coordinateOffset] using hinner

theorem contiguous_vector_offset (tensor : Tensor) (N : Nat) (hs : tensor.shape=[N])
    (hd : ContiguousLayout tensor) (index : Nat) (hi : index<N) :
    coordinateOffset tensor.strides [index]=index := by
  have he := hd [index] (coordinate_vector tensor N index hs hi)
  simpa [hs,rowMajorOffset] using he

theorem contiguous_matrix_offset (tensor : Tensor) (M N : Nat) (hs : tensor.shape=[M,N])
    (hd : ContiguousLayout tensor) (row col : Nat) (hr : row<M) (hc : col<N) :
    coordinateOffset tensor.strides [row,col]=row*N+col := by
  have he := hd [row,col] (coordinate_matrix tensor M N row col hs hr hc)
  simpa [hs,rowMajorOffset] using he

theorem materialize_inner (api : TensorAPI world) (hl : LayoutAPI api)
    (target : Nat) (role : Role) (input : Tensor) (s : S) (output : Tensor)
    (h : ((materialize api target role input).run s).result=.ok output)
    (hi : input.strides.getLast?=some 1) : InnerDense output := by
  unfold materialize at h
  dsimp only [Bind.bind] at h
  obtain ⟨resolved,hr,ht,_⟩ := bind_success _ _ _ _ h
  exact hl.transfer_inner target role resolved _ output (call_success _ _ _ _ ht)
    (hl.resolve_inner role input s resolved (call_success _ _ _ _ hr) hi)

theorem materialize_contiguous (api : TensorAPI world) (hl : LayoutAPI api)
    (target : Nat) (role : Role) (input : Tensor) (s : S) (output : Tensor)
    (h : ((materialize api target role input).run s).result=.ok output)
    (hi : input.contiguous=Bool.true) : ContiguousLayout output := by
  unfold materialize at h
  dsimp only [Bind.bind] at h
  obtain ⟨resolved,hr,ht,_⟩ := bind_success _ _ _ _ h
  exact hl.transfer_contiguous target role resolved _ output (call_success _ _ _ _ ht)
    (hl.resolve_contiguous role input s resolved (call_success _ _ _ _ hr) hi)

theorem resolveOption_property (api : TensorAPI world) (target : Nat) (role : Role) (input : Option Tensor)
    (P : Tensor → Prop) (s : S) (output : Option Tensor)
    (h : ((resolveOption api target role input).run s).result=.ok output)
    (hp : ∀ tensor, input=some tensor → ∀ current value,
      ((materialize api target role tensor).run current).result=.ok value → P value) : OptionalFacts output P := by
  cases input with
  | none =>
    have he : output=none := by simpa [resolveOption,Pure.pure,Action.pure] using h.symm
    simp [OptionalFacts,he]
  | some tensor =>
    unfold resolveOption at h
    dsimp only [Bind.bind] at h
    obtain ⟨value,hv,h,_⟩ := bind_success _ _ _ _ h
    have he : output=some value := by simpa [Pure.pure,Action.pure] using h.symm
    intro out hout
    have ho : out=value := Option.some.inj (hout.symm.trans he)
    rw [ho]
    exact hp tensor rfl s value hv

structure InputsDense (input : Inputs) : Prop where
  x : InnerDense input.x
  weight : InnerDense input.weight
  bias : OptionalFacts input.bias InnerDense
  residual : OptionalFacts input.residual InnerDense
  x1 : OptionalFacts input.x1 InnerDense
  weight1 : OptionalFacts input.weight1 InnerDense
  bias1 : OptionalFacts input.bias1 InnerDense
  rowscale : OptionalFacts input.rowscale ContiguousLayout

/-- The original validation is consumed before each actual resolve call. The
result speaks about immutable metadata of the handles actually returned. -/
theorem resolveInputs_dense (api : TensorAPI world) (hl : LayoutAPI api)
    (q : Request) (v : Validated) (hv : ValidatedFacts q v) (s : S) (input : Inputs)
    (h : ((resolveInputs api q).run s).result=.ok input) : InputsDense input := by
  unfold resolveInputs at h
  dsimp only [Bind.bind] at h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x,hx,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨weight,hw,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨bias,hb,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨residual,hr,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x1,hx1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨weight1,hw1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨bias1,hb1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨rowscale,hrs,h,_⟩ := bind_success _ _ _ _ h
  have he : input=⟨q.x,x,weight,bias,residual,x1,weight1,bias1,rowscale⟩ := by
    simpa [Pure.pure,Action.pure] using h.symm
  rw [he]
  refine ⟨materialize_inner api hl q.x.device .x q.x _ x hx hv.x.2,
    materialize_inner api hl q.x.device .weight q.weight _ weight hw hv.weight.2,?_,?_,?_,?_,?_,?_⟩
  · refine resolveOption_property api q.x.device .bias q.bias InnerDense _ bias hb ?_
    intro tensor ht current value hs
    exact materialize_inner api hl q.x.device .bias tensor current value hs (hv.bias tensor ht).2
  · refine resolveOption_property api q.x.device .residual q.residual InnerDense _ residual hr ?_
    intro tensor ht current value hs
    exact materialize_inner api hl q.x.device .residual tensor current value hs (hv.residual tensor ht).2
  · refine resolveOption_property api q.x.device .x1 q.x1 InnerDense _ x1 hx1 ?_
    intro tensor ht current value hs
    exact materialize_inner api hl q.x.device .x1 tensor current value hs (hv.x1 tensor ht).2.2
  · refine resolveOption_property api q.x.device .weight1 q.weight1 InnerDense _ weight1 hw1 ?_
    intro tensor ht current value hs
    exact materialize_inner api hl q.x.device .weight1 tensor current value hs (hv.weight1 tensor ht).2
  · refine resolveOption_property api q.x.device .bias1 q.bias1 InnerDense _ bias1 hb1 ?_
    intro tensor ht current value hs
    exact materialize_inner api hl q.x.device .bias1 tensor current value hs (hv.bias1 tensor ht).2
  · refine resolveOption_property api q.x.device .rowscale q.rowscale ContiguousLayout _ rowscale hrs ?_
    intro tensor ht current value hs
    exact materialize_contiguous api hl q.x.device .rowscale tensor current value hs (hv.rowscale tensor ht).2

#axiomsClean innerDense_of_last
#axiomsClean matrix_active_offset
#axiomsClean vector_active_offset
#axiomsClean contiguous_vector_offset
#axiomsClean contiguous_matrix_offset
#axiomsClean materialize_inner
#axiomsClean materialize_contiguous
#axiomsClean resolveInputs_dense
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
