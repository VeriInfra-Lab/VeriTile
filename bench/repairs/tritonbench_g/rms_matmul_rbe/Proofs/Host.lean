import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

inductive DType where
  | fp32 | fp16 | int8
  deriving DecidableEq
structure Tensor where
  storage : Nat
  offset : Nat
  shape : List Nat
  strides : List Nat
  dtype : DType
  device : Nat
  cuda : Bool
  ordinary : Bool
  extent : Nat
def weightDType (b : Buffers) := if b.useByte then DType.int8 else DType.fp16
def outputDType (b : Buffers) (phase : Fin 3) :=
  if phase.val=0 then DType.fp16 else weightDType b
def tensor3 (v : View3) (s0 s1 s2 : Nat) (dtype : DType) (device extent : Nat) : Tensor :=
  ⟨v.storage,v.offset,[s0,s1,s2],[v.sb,v.sm,v.sn],dtype,device,Bool.true,Bool.true,extent⟩
def returned (s : Shape) (b : Buffers) (device : Nat) (extents : Nat → Nat) (phase : Fin 3) : Tensor :=
  let v := b.output phase
  ⟨v.storage,v.offset,[s.B,s.M,s.heads,s.headDim],
    [v.sb,v.sm,s.headDim*v.sn,v.sn],outputDType b phase,device,Bool.true,Bool.true,extents v.storage⟩

/-- Ordinary Tensor.view contract: compatible layout, no copy, same storage,
offset, dtype, device, and allocation. No value equation occurs here. -/
structure ViewABI (before after : Tensor) : Prop where
  storage : after.storage=before.storage
  offset : after.offset=before.offset
  dtype : after.dtype=before.dtype
  device : after.device=before.device
  allocation : after.extent=before.extent
  elementCount : after.shape.prod=before.shape.prod
  ordinary : before.ordinary=Bool.true ∧ after.ordinary=Bool.true
  cuda : before.cuda=Bool.true ∧ after.cuda=Bool.true

structure HostDomain (s : Shape) (p : Parameters F) (b : Buffers) where
  valid : s.Valid
  parameterK : p.K=s.K
  parameterHead : p.headDim=s.headDim
  device : Nat
  extents : Nat → Nat
  input : Tensor
  weights : Fin 3 → Tensor
  rmsTensor : Tensor
  inputMetadata : input=tensor3 b.x s.B s.M s.K DType.fp32 device (extents b.x.storage)
  weightMetadata : ∀ phase,weights phase=
    ⟨(b.weight phase).storage,(b.weight phase).offset,[s.N,s.K],
      [(b.weight phase).sn,(b.weight phase).sk],weightDType b,device,Bool.true,Bool.true,
      extents (b.weight phase).storage⟩
  rmsMetadata : rmsTensor=⟨b.rms.storage,b.rms.offset,[s.K],[b.rms.stride],
    weightDType b,device,Bool.true,Bool.true,extents b.rms.storage⟩
  suppliedK : Tensor
  suppliedV : Tensor
  kView : ViewABI suppliedK (tensor3 (b.output 1) s.B s.M s.N (weightDType b) device (extents (b.output 1).storage))
  vView : ViewABI suppliedV (tensor3 (b.output 2) s.B s.M s.N (weightDType b) device (extents (b.output 2).storage))
  qDense : (b.output 0).offset=0 ∧ (b.output 0).sn=1 ∧ (b.output 0).sm=s.N ∧ (b.output 0).sb=(max s.M 1)*s.N
  qExtent : extents (b.output 0).storage=s.B*s.M*s.N
  separated : b.Separate
  nonoverlap : Function.Injective (outputAddress b : Cell s → Address)
  indices : IndexDomain s b p.start
  xAllocated : ∀ batch row k,batch<s.B → row<s.M → k<s.K → (b.x.address batch row k).2<extents b.x.storage
  rmsAllocated : ∀ k,k<s.K → (b.rms.address k).2<extents b.rms.storage
  weightAllocated : ∀ phase n k,n<s.N → k<s.K → ((b.weight phase).address n k).2<extents (b.weight phase).storage
  outputAllocated : ∀ x : Cell s,(outputAddress b x).2<extents (outputAddress b x).1
  epsilonReal : ℝ
  thetaReal : ℝ
  epsilonPositive : 0<epsilonReal
  thetaPositive : 0<thetaReal

abbrev shouldLaunch (s : Shape) : Prop := 0<s.B ∧ 0<s.M ∧ 0<s.N
def hostCode (s : Shape) (binding : CompilerBinding F H s p b) : Code F H Unit :=
  if shouldLaunch s then binding.program else pure ()
theorem schedule_empty (schedule : Schedule s) (empty : ¬shouldLaunch s) : schedule.events=[] := by
  by_contra h
  cases he : schedule.events with
  | nil => exact h he
  | cons x xs =>
    have hb := x.2.1.isLt
    have hm := x.2.2.1.isLt
    have hn := x.2.2.2.isLt
    exact empty ⟨by omega,by omega,by omega⟩
theorem hostCode_eq (binding : CompilerBinding F H s p b) : hostCode s binding=binding.program := by
  by_cases h : shouldLaunch s
  · simp [hostCode,h]
  · rw [hostCode,if_neg h,binding.lowered,schedule_empty binding.schedule h]
    rfl

theorem return_views (domain : HostDomain s p b) :
    (returned s b domain.device domain.extents 1).storage=domain.suppliedK.storage ∧
    (returned s b domain.device domain.extents 2).storage=domain.suppliedV.storage ∧
    (∀ phase,(returned s b domain.device domain.extents phase).shape=[s.B,s.M,s.heads,s.headDim]) ∧
    (returned s b domain.device domain.extents 0).dtype=DType.fp16 := by
  exact ⟨domain.kView.storage,domain.vView.storage,fun _ => rfl,rfl⟩

/-- The state is after torch.empty allocates fresh dense Q. The allocation ABI
preserves all caller storage except that new allocation; Tensor.view preserves
the supplied K/V storage and offsets. Returned objects need not be identical. -/
theorem whole_host
    (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (allocationFrame : ∀ q,q.1≠(b.output 0).storage → abi.observe state q=caller q) :
    (∀ x : Cell s,abi.observe ((hostCode s binding).native abi state).2 (outputAddress b x)=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠(b.output 0).storage → (∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=caller q) ∧
    (returned s b domain.device domain.extents 1).storage=domain.suppliedK.storage ∧
    (returned s b domain.device domain.extents 2).storage=domain.suppliedV.storage := by
  rw [hostCode_eq]
  have h := bound_native_complete actual reference leaves abi p b domain.separated domain.nonoverlap binding state
  refine ⟨h.1,h.2,?_,domain.kView.storage,domain.vView.storage⟩
  intro q fresh outside
  rw [show abi.observe (binding.program.native abi state).2 q=abi.observe state q from h.2 q outside]
  exact allocationFrame q fresh

#axiomsClean schedule_empty
#axiomsClean hostCode_eq
#axiomsClean return_views
#axiomsClean whole_host
end
end VeriTile.Bench.Repairs.RmsMatmulRbe
