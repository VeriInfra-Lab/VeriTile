import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000

inductive DType where
  | fp16 | int8
  deriving DecidableEq
structure Tensor where
  storage : Nat
  offset : Nat
  shape : List Nat
  strides : List Nat
  dtype : DType
  device : Nat
  extent : Nat
def weightDType (b : Buffers) := if b.useByte then DType.int8 else DType.fp16

/-- The proof's singleton outer coordinate is not a GPU batch axis. The actual
kernel flattens batch and sequence into its single M dimension. -/
structure HostDomain (s : Shape) (p : Parameters (BitVec 32)) (b : Buffers) where
  valid : s.Valid
  parameterK : p.K=s.K
  epsilon : p.epsilon=BitVec.ofNat 32 0x358637bd
  batch : Nat
  seq : Nat
  flatten : s.M=batch*seq
  device : Nat
  extents : Nat → Nat
  input : Tensor
  w1 : Tensor
  w3 : Tensor
  rms : Tensor
  inputMetadata : input=⟨b.x.storage,0,[batch,seq,s.K],input.strides,.fp16,device,extents b.x.storage⟩
  w1Metadata : w1=⟨b.weight.storage,0,[s.N,s.K],w1.strides,weightDType b,device,extents b.weight.storage⟩
  w3Metadata : w3=⟨b.weight3.storage,0,[s.N,s.K],w3.strides,weightDType b,device,extents b.weight3.storage⟩
  rmsMetadata : rms=⟨b.rms.storage,0,[s.K],rms.strides,weightDType b,device,extents b.rms.storage⟩
  xDense : b.x.offset=0 ∧ b.x.sm=s.K ∧ b.x.sn=1
  w1Dense : b.weight.offset=0 ∧ b.weight.sn=s.K ∧ b.weight.sk=1
  w3Dense : b.weight3.offset=0 ∧ b.weight3.sn=s.K ∧ b.weight3.sk=1
  rmsDense : b.rms.offset=0 ∧ b.rms.stride=1
  outputDense : b.output.offset=0 ∧ b.output.sm=s.N ∧ b.output.sn=1
  fresh : b.Separate
  nonoverlap : Function.Injective (outputAddress b : Cell s → Address)
  indices : 0<s.M → IndexDomain s b
  xAllocated : ∀ row k,row<s.M → k<s.K → (b.x.address 0 row k).2<extents b.x.storage
  rmsAllocated : ∀ k,k<s.K → (b.rms.address k).2<extents b.rms.storage
  w1Allocated : ∀ n k,n<s.N → k<s.K → (b.weight.address n k).2<extents b.weight.storage
  w3Allocated : ∀ n k,n<s.N → k<s.K → (b.weight3.address n k).2<extents b.weight3.storage
  outputAllocated : ∀ x : Cell s,(outputAddress b x).2<extents b.output.storage

def returned (s : Shape) (b : Buffers) (batch seq device : Nat) (extents : Nat → Nat) : Tensor :=
  ⟨b.output.storage,0,[batch,seq,s.N],[seq*s.N,s.N,1],.fp16,device,extents b.output.storage⟩
def logicalCell (domain : HostDomain s p b) (batch : Fin domain.batch) (seq : Fin domain.seq)
    (n : Fin s.N) : Cell s :=
  (⟨0,by have h:=domain.valid.1; omega⟩,
   ⟨batch.val*domain.seq+seq.val,by
     have hb:=batch.isLt
     have hs:=seq.isLt
     have h:=Nat.mul_le_mul_right domain.seq (Nat.succ_le_of_lt hb)
     rw [domain.flatten]
     nlinarith⟩,n)
theorem logical_input_address (domain : HostDomain s p b) (batch : Fin domain.batch)
    (seq : Fin domain.seq) (k : Nat) :
    b.x.address 0 (batch.val*domain.seq+seq.val) k=
      (b.x.storage,batch.val*(domain.seq*s.K)+seq.val*s.K+k) := by
  simp only [View3.address,domain.xDense.1,domain.xDense.2.1,domain.xDense.2.2]
  congr 1
  ring
theorem logical_output_address (domain : HostDomain s p b) (batch : Fin domain.batch)
    (seq : Fin domain.seq) (n : Fin s.N) :
    outputAddress b (logicalCell domain batch seq n)=
      (b.output.storage,batch.val*(domain.seq*s.N)+seq.val*s.N+n.val) := by
  simp only [outputAddress,logicalCell,View3.address,domain.outputDense.1,
    domain.outputDense.2.1,domain.outputDense.2.2]
  congr 1
  ring

theorem schedule_empty (schedule : Schedule s) (empty : ¬shouldLaunch s) : schedule.events=[] := by
  by_contra h
  cases he : schedule.events with
  | nil => exact h he
  | cons x xs => have hm:=x.2.1.isLt; exact empty (by unfold shouldLaunch; omega)
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
theorem returned_shape (domain : HostDomain s p b) :
    (returned s b domain.batch domain.seq domain.device domain.extents).storage=b.output.storage ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).shape=[domain.batch,domain.seq,s.N] ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).dtype=DType.fp16 ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).shape.prod=s.M*s.N := by
  simp [returned,domain.flatten,Nat.mul_assoc]

/-- Allocation is the ordinary host boundary; every subsequent output and frame
equation follows from the current-memory source execution proof. -/
theorem whole_host (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (allocationFrame : ∀ q,q.1≠b.output.storage → abi.observe state q=caller q) :
    (∀ x : Cell s,abi.observe ((hostCode s binding).native abi state).2 (outputAddress b x)=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((hostCode s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠b.output.storage → abi.observe ((hostCode s binding).native abi state).2 q=caller q) := by
  rw [hostCode_eq]
  have h := bound_native_complete actual reference leaves abi p b domain.fresh domain.nonoverlap binding state
  refine ⟨h.1,h.2,?_⟩
  intro q outside
  rw [show abi.observe (binding.program.native abi state).2 q=abi.observe state q from
    h.2 q (fun x he => outside (congrArg Prod.fst he))]
  exact allocationFrame q outside

#axiomsClean logical_input_address
#axiomsClean logical_output_address
#axiomsClean hostCode_eq
#axiomsClean device_restored
#axiomsClean returned_shape
#axiomsClean whole_host
end
end VeriTile.Bench.Repairs.LlamaFF
