import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorage

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

def OptionalStrideBound (input : Option Tensor) (value : Int) : Prop :=
  match input with | none => value=0 | some tensor => tensor.strides[0]?=some value

structure ArgumentsBound (args : LaunchArguments) (p : Prepared) (q : Request) : Prop where
  prepared : args.prepared=p
  request : args.request=q
  x : p.inputs.x.strides[0]?=some args.strideX
  y : p.outputs.y.strides[0]?=some args.strideY
  residual : OptionalStrideBound p.inputs.residual args.strideResidual
  residualOut : OptionalStrideBound p.outputs.residualOut args.strideResidualOut
  x1 : OptionalStrideBound p.inputs.x1 args.strideX1
  y1 : OptionalStrideBound p.outputs.y1 args.strideY1

theorem stride_zero_success (role : Role) (tensor : Tensor) (s : S) (value : Int)
    (h : ((stride (world:=world) role tensor 0).run s).result=.ok value) : tensor.strides[0]?=some value := by
  cases he : tensor.strides[0]? with
  | none => simp [stride,checked,he] at h
  | some v =>
    have hv : v=value := by simpa [stride,checked,he] using h
    rw [hv]

theorem optional_stride_success (role : Role) (input : Option Tensor) (s : S) (value : Int)
    (h : ((optionalStride (world:=world) role input).run s).result=.ok value) : OptionalStrideBound input value := by
  cases input with
  | none => simpa [optionalStride,Pure.pure,Action.pure,OptionalStrideBound] using h.symm
  | some tensor => exact stride_zero_success role tensor s value h

/-- Successful launch arguments are the actual post-materialization tensor
strides in source order, with zero only for absent optional handles. -/
theorem arguments_bound (api : DispatchAPI world) (q : Request) (p : Prepared) (s : S) (args : LaunchArguments)
    (h : ((arguments api q p).run s).result=.ok args) : ArgumentsBound args p q := by
  unfold arguments at h
  dsimp only [Bind.bind] at h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨sx,hx,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨sy,hy,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨sr,hr,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨so,ho,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨sx1,hx1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨sy1,hy1,h,_⟩ := bind_success _ _ _ _ h
  have ha : args=⟨p,q,sx,sy,sr,so,sx1,sy1⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [ha]
  exact ⟨rfl,rfl,stride_zero_success .x p.inputs.x _ sx hx,stride_zero_success .y p.outputs.y _ sy hy,
    optional_stride_success .residual p.inputs.residual _ sr hr,
    optional_stride_success .residualOut p.outputs.residualOut _ so ho,
    optional_stride_success .x1 p.inputs.x1 _ sx1 hx1,optional_stride_success .y1 p.outputs.y1 _ sy1 hy1⟩

theorem persists_optionalStride (store : TensorStore S Descriptor Value) (role : Role) (input : Option Tensor) :
    Persists (world:=world) store (optionalStride role input) := by
  cases input
  · exact persists_pure store 0
  · exact persists_stride store _ _ _

theorem persists_arguments (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hc : DispatchStorage store api) (q : Request) (p : Prepared) : Persists store (arguments api q p) := by
  unfold arguments
  apply persists_bind store _ _ (persists_call store _ _ (hc.select _))
  intro _
  apply persists_bind store _ _ (persists_stride store _ _ _)
  intro _
  apply persists_bind store _ _ (persists_stride store _ _ _)
  intro _
  apply persists_bind store _ _ (persists_optionalStride store _ _)
  intro _
  apply persists_bind store _ _ (persists_optionalStride store _ _)
  intro _
  apply persists_bind store _ _ (persists_optionalStride store _ _)
  intro _
  apply persists_bind store _ _ (persists_optionalStride store _ _)
  intro _
  exact persists_pure store _

#axiomsClean stride_zero_success
#axiomsClean optional_stride_success
#axiomsClean arguments_bound
#axiomsClean persists_arguments
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
