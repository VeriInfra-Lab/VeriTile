import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Bench.Repairs.RmsMatmulRbe
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
def tensor3 (v : View3) (s0 s1 s2 : Nat) (dtype : DType) (device extent : Nat) : Tensor :=
  ⟨v.storage,v.offset,[s0,s1,s2],[v.sb,v.sm,v.sn],dtype,device,Bool.true,Bool.true,extent⟩
def returned (s : Shape) (b : Buffers) (device : Nat) (extents : Nat → Nat) : Tensor :=
  let v := b.output
  ⟨v.storage,v.offset,[s.B,s.M,s.heads,s.headDim],
    [v.sb,v.sm,s.headDim*v.sn,v.sn],weightDType b,device,Bool.true,Bool.true,extents v.storage⟩
def transposed (v : View2) (s : Shape) (b : Buffers) (device extent : Nat) : Tensor :=
  ⟨v.storage,v.offset,[s.K,s.N],[v.sk,v.sn],weightDType b,device,Bool.true,Bool.true,extent⟩

/-- Tensor.to performs one ordinary numeric dtype conversion per element.
This scalar operation converts to signed int8 when requested; E5M2 payload
reinterpretation occurs later in loadRms. The two operations are distinct. -/
abbrev HostCast (F H : Type) := DType → DType → Stored F H → Stored F H
def castWord (cast : HostCast F H) (src dst : DType) (v : Stored F H) : Stored F H :=
  if src=dst then v else cast src dst v
structure HostCastLeaf (actual reference : HostCast (BitVec 32) (BitVec 16)) : Prop where
  scalarCast : ∀ src dst word,actual src dst word=reference src dst word
theorem castWord_reference (a r : HostCast (BitVec 32) (BitVec 16)) (h : HostCastLeaf a r)
    (src dst : DType) (v : Stored (BitVec 32) (BitVec 16)) :
    castWord a src dst v=castWord r src dst v := by
  simp only [castWord,h.scalarCast]

def rmsFromWord (a : StorageOps F H) (byte : Bool) (v : Stored F H) : F :=
  loadRms a byte (fun _ => v) (0,0)
def hostRowInput (a : StorageOps F H) (cast : HostCast F H) (src : DType)
    (originalRms : View1) (b : Buffers) (caller : Memory F H) (batch row : Nat) : RowInput F where
  x := fun k => read32 a caller (b.x.address batch row k)
  rms := fun k => rmsFromWord a b.useByte (castWord cast src (weightDType b) (caller (originalRms.address k)))
  weight := fun n k => loadWeight a b.useByte caller (b.weight.address n k)
def hostTarget (a : StorageOps F H) (cast : HostCast F H) (src : DType)
    (originalRms : View1) (p : Parameters F) (b : Buffers) (caller : Memory F H) (x : Cell s) : Stored F H :=
  encode a b (referenceValue a.toOps p (hostRowInput a cast src originalRms b caller x.1 x.2.1)
    (selectedPlans x.1 x.2.1) (selectedReductions x.1 x.2.1) b.useRotary x.2.1 x.2.2)

/-- Ordinary preparation ABI: tensor allocation/frame, per-scalar Tensor.to,
and readonly input preservation. No dot, reduction, rotary or output result law. -/
structure Preparation (K : Nat) (cast : HostCast F H) (src : DType) (originalRms : View1)
    (b : Buffers) (caller prepared : Memory F H) : Prop where
  x : ∀ batch row k,prepared (b.x.address batch row k)=caller (b.x.address batch row k)
  weight : ∀ n k,prepared (b.weight.address n k)=caller (b.weight.address n k)
  rms : ∀ k,k<K → prepared (b.rms.address k)=castWord cast src (weightDType b) (caller (originalRms.address k))
  noCastAlias : src=weightDType b → b.rms=originalRms
  castFresh : src≠weightDType b → b.rms.storage≠originalRms.storage
  frame : ∀ q,q.1≠b.output.storage → (src=weightDType b ∨ q.1≠b.rms.storage) → prepared q=caller q

theorem reference_rms_congr (a : Ops F) (p : Parameters F) (x y : RowInput F)
    (hx : x.x=y.x) (hw : x.weight=y.weight) (hr : ∀ k,k<p.K → x.rms k=y.rms k)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) :
    referenceValue a p x plans reduction rotary row n=referenceValue a p y plans reduction rotary row n := by
  have hm : masked a p x.rms=masked a p y.rms := by
    funext t l
    unfold masked
    split_ifs with h
    · exact hr _ h
    · rfl
  have moments : referenceMoment a p x=referenceMoment a p y := by
    funext l
    simp only [referenceMoment,hx]
  simp only [referenceValue,referenceProjection,referenceAccumulation,referenceNorm,
    moments,instruction,scaled,hx,hw,hm]

theorem prepared_target (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (castA castR : HostCast (BitVec 32) (BitVec 16)) (castLeaf : HostCastLeaf castA castR)
    (src : DType) (originalRms : View1) (p : Parameters (BitVec 32)) (b : Buffers)
    (caller prepared : Memory (BitVec 32) (BitVec 16))
    (prep : Preparation p.K castA src originalRms b caller prepared) (x : Cell s) :
    wordTarget reference p b selectedPlans selectedReductions prepared x=
      hostTarget reference castR src originalRms p b caller x := by
  unfold wordTarget hostTarget
  congr 1
  apply reference_rms_congr
  · funext k
    simp only [rowInput,hostRowInput,read32,prep.x]
  · funext n k
    simp only [rowInput,hostRowInput,loadWeight,read8,read16,prep.weight]
  · intro k hk
    simp only [rowInput,hostRowInput,rmsFromWord,loadRms,read8,read16,prep.rms k hk,
      castWord_reference castA castR castLeaf]

structure HostDomain (s : Shape) (p : Parameters F) (b : Buffers) where
  valid : s.Valid
  parameterK : p.K=s.K
  parameterHead : p.headDim=s.headDim
  device : Nat
  extents : Nat → Nat
  input : Tensor
  weight : Tensor
  rmsTensor : Tensor
  originalRms : View1
  sourceDType : DType
  inputMetadata : input=tensor3 b.x s.B s.M s.K DType.fp32 device (extents b.x.storage)
  weightMetadata : weight=⟨b.weight.storage,b.weight.offset,[s.N,s.K],
    [b.weight.sn,b.weight.sk],weightDType b,device,Bool.true,Bool.true,extents b.weight.storage⟩
  rmsMetadata : rmsTensor=⟨originalRms.storage,originalRms.offset,[s.K],[originalRms.stride],
    sourceDType,device,Bool.true,Bool.true,extents originalRms.storage⟩
  dense : b.output.offset=0 ∧ b.output.sn=1 ∧ b.output.sm=s.N ∧ b.output.sb=(max s.M 1)*s.N
  outExtent : extents b.output.storage=s.B*s.M*s.N
  separated : b.Separate
  nonoverlap : Function.Injective (outputAddress b : Cell s → Address)
  indices : (0<s.B ∧ 0<s.M) → IndexDomain s b p.start
  xAllocated : ∀ batch row k,batch<s.B → row<s.M → k<s.K → (b.x.address batch row k).2<extents b.x.storage
  rmsAllocated : ∀ k,k<s.K → (b.rms.address k).2<extents b.rms.storage
  weightAllocated : ∀ n k,n<s.N → k<s.K → (b.weight.address n k).2<extents b.weight.storage
  outputAllocated : ∀ x : Cell s,(outputAddress b x).2<extents (outputAddress b x).1

abbrev shouldLaunch (s : Shape) : Prop := 0<s.B ∧ 0<s.M ∧ 0<s.N
def hostCode (s : Shape) (binding : CompilerBinding F H s p b) : Code F H Unit :=
  if shouldLaunch s then binding.program else pure ()
theorem schedule_empty (schedule : Schedule s) (empty : ¬shouldLaunch s) : schedule.events=[] := by
  by_contra h
  cases he : schedule.events with
  | nil => exact h he
  | cons x xs =>
    have hb := x.1.isLt
    have hm := x.2.1.isLt
    have hn := x.2.2.isLt
    exact empty ⟨by omega,by omega,by omega⟩
theorem hostCode_eq (binding : CompilerBinding F H s p b) : hostCode s binding=binding.program := by
  by_cases h : shouldLaunch s
  · simp [hostCode,h]
  · rw [hostCode,if_neg h,binding.lowered,schedule_empty binding.schedule h]
    rfl

def withDevice (device : Nat) (body : Nat × σ → Nat × σ) (before : Nat × σ) : Nat × σ :=
  let after := body (device,before.2)
  (before.1,after.2)
theorem device_restored (device : Nat) (body : Nat × σ → Nat × σ) (before : Nat × σ) :
    (withDevice device body before).1=before.1 := rfl

theorem return_view (domain : HostDomain s p b) :
    (returned s b domain.device domain.extents).storage=b.output.storage ∧
    (returned s b domain.device domain.extents).shape=[s.B,s.M,s.heads,s.headDim] ∧
    (returned s b domain.device domain.extents).dtype=weightDType b ∧
    (returned s b domain.device domain.extents).shape.prod=s.B*s.M*s.N := by
  refine ⟨rfl,rfl,rfl,?_⟩
  simp only [returned,List.prod_cons,List.prod_nil,Nat.mul_one]
  rw [domain.valid.2.2.2.2]
  ac_rfl

theorem transpose_address (v : View2) (s : Shape) (b : Buffers) (device extent n k : Nat) :
    (transposed v s b device extent).storage=v.storage ∧
    v.offset+k*v.sk+n*v.sn=(v.address n k).2 := by
  constructor
  · rfl
  · simp only [View2.address]; omega

theorem whole_host
    (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (castA castR : HostCast (BitVec 32) (BitVec 16)) (castLeaf : HostCastLeaf castA castR)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (prep : Preparation p.K castA domain.sourceDType domain.originalRms b caller (abi.observe state)) :
    (∀ x : Cell s,abi.observe ((hostCode s binding).native abi state).2 (outputAddress b x)=
      hostTarget reference castR domain.sourceDType domain.originalRms p b caller x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠b.output.storage → (domain.sourceDType=weightDType b ∨ q.1≠b.rms.storage) →
      abi.observe ((hostCode s binding).native abi state).2 q=caller q) := by
  rw [hostCode_eq]
  have h := bound_native_complete actual reference leaves abi p b domain.separated domain.nonoverlap binding state
  refine ⟨?_,h.2,?_⟩
  · intro x
    apply (h.1 x).trans
    exact prepared_target actual reference castA castR castLeaf domain.sourceDType domain.originalRms p b caller (abi.observe state) prep x
  · intro q fresh original
    rw [show abi.observe (binding.program.native abi state).2 q=abi.observe state q from h.2 q (fun x he => fresh (congrArg Prod.fst he))]
    exact prep.frame q fresh original

#axiomsClean castWord_reference
#axiomsClean reference_rms_congr
#axiomsClean prepared_target
#axiomsClean schedule_empty
#axiomsClean hostCode_eq
#axiomsClean device_restored
#axiomsClean return_view
#axiomsClean transpose_address
#axiomsClean whole_host
end
end VeriTile.Bench.Repairs.RmsRbeMatmul
