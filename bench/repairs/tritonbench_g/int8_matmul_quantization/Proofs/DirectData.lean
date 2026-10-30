import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Contract
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Domain

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

/-- Original object, exact strides and physical logical-element map. -/
structure TensorView where
  info : TensorInfo
  objectID : Nat
  strides : List Nat
  physicalIndex : Nat → Nat

def logicalAddress (t : TensorView) (i : Nat) : Address := (t.info.storage,t.physicalIndex i)
def nativePointer (ptr : BitVec 64) (bytes offset : Nat) : Nat :=
  (ptr+BitVec.ofNat 64 (bytes*offset)).toNat
structure TensorLaws (base : Nat → Nat) (t : TensorView) : Prop where
  pointer : 0<t.info.metadata.numel →
    t.info.metadata.dataPtr=BitVec.ofNat 64 (base t.info.storage+bytes t.info*t.info.offset)
  contiguous : t.info.metadata.isContiguous=Bool.true → ∀ i,i<t.info.metadata.numel →
    t.physicalIndex i=t.info.offset+i
  column : t.info.transposeContiguous=Bool.true → ∀ row col,row<rows t.info → col<cols t.info →
    t.physicalIndex (row*cols t.info+col)=t.info.offset+col*rows t.info+row
  capacity : 0<t.info.metadata.numel → base t.info.storage+bytes t.info*(t.info.offset+t.info.metadata.numel)≤2^64

theorem tensor_bytes_positive (t : TensorInfo) : 0<bytes t := by
  unfold bytes;split_ifs <;> decide
theorem original_pointer (base : Nat → Nat) (t : TensorView) (laws : TensorLaws base t)
    (delta : Nat) (bound : delta<t.info.metadata.numel) :
    nativePointer t.info.metadata.dataPtr (bytes t.info) delta=
      base t.info.storage+bytes t.info*(t.info.offset+delta) := by
  have nonempty : 0<t.info.metadata.numel:=by omega
  have cap:=laws.capacity nonempty
  have lt:=Nat.mul_lt_mul_of_pos_left
    (show t.info.offset+delta<t.info.offset+t.info.metadata.numel by omega) (tensor_bytes_positive t.info)
  have native : base t.info.storage+bytes t.info*(t.info.offset+delta)<2^64:=by omega
  simp only [nativePointer,laws.pointer nonempty,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show base t.info.storage+bytes t.info*t.info.offset+bytes t.info*delta=
    base t.info.storage+bytes t.info*(t.info.offset+delta) by ring]
  exact Nat.mod_eq_of_lt native
theorem original_read (base : Nat → Nat) (t : TensorView) (laws : TensorLaws base t)
    (contiguous : t.info.metadata.isContiguous=Bool.true) (m : Memory F H)
    (i : Nat) (hi : i<t.info.metadata.numel) : m (t.info.storage,t.info.offset+i)=m (logicalAddress t i) := by
  simp only [logicalAddress,laws.contiguous contiguous i hi]
theorem matrix_shape (t : TensorInfo) (rank : t.metadata.shape.length=2) : t.metadata.shape=[rows t,cols t] := by
  rcases t with ⟨⟨dtype,layout,device,shape,contiguous,neg,pointer⟩,storage,offset,trans⟩
  rcases shape with _|⟨r,tail⟩
  · simp at rank
  rcases tail with _|⟨c,tail⟩
  · simp at rank
  cases tail with
  | nil=>rfl
  | cons _ _=>simp at rank
theorem matrix_numel (t : TensorInfo) (rank : t.metadata.shape.length=2) : t.metadata.numel=rows t*cols t := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [matrix_shape t rank]
  simp
theorem row_contiguous (h : TensorContract t types rank Bool.false) : t.metadata.isContiguous=Bool.true := by
  rcases h.dense with h|h
  · exact h
  · simp at h

structure GemmRequest where
  a : TensorView
  aScale : TensorView
  b : TensorView
  bScale : TensorView
  out : Option TensorView
  contract : GemmContract a.info (some aScale.info) b.info bScale.info (out.map TensorView.info) Bool.false
def GemmRequest.shape (r : GemmRequest) := Direct.shape r.a.info r.b.info
def GemmRequest.buffers (r : GemmRequest) (out : Nat) :=
  gemmBuffers r.a.info (some r.aScale.info) r.b.info r.bScale.info (r.out.map TensorView.info) Bool.false 0 0 out
def GemmRequest.owner (r : GemmRequest) := r.a.info.metadata.device.index
def GemmRequest.views (r : GemmRequest) := [r.a,r.aScale,r.b,r.bScale]
def GemmRequest.live (r : GemmRequest) (live : Nat → Prop) : Prop := ∀ t∈r.views,live t.info.storage
structure GemmLaws (base : Nat → Nat) (r : GemmRequest) : Prop where
  inputs : ∀ t∈r.views,TensorLaws base t
  output : ∀ t,r.out=some t → TensorLaws base t

/-- Successful empty/zeros allocation descriptor, independent of arithmetic. -/
def FreshDescriptor (t : TensorView) (dtype : VeriTile.Torch.DType) (dims strides : List Nat)
    (device : VeriTile.Torch.Device) : Prop :=
  t.info.metadata=⟨dtype,.strided,device,dims,Bool.true,Bool.false,t.info.metadata.dataPtr⟩ ∧
  t.info.offset=0 ∧ t.strides=strides
def GemmDescriptor (r : GemmRequest) (t : TensorView) : Prop :=
  match r.out with
  | some supplied=>t=supplied
  | none=>FreshDescriptor t (.other "float16") [r.shape.M,r.shape.N] [max r.shape.N 1,1] r.a.info.metadata.device

structure GemmAllocation (r : GemmRequest) (caller : Memory F H) (live : Nat → Prop) where
  output : TensorView
  memory : Memory F H
  descriptor : GemmDescriptor r output
  fresh : r.out=none → ¬live output.info.storage
  frame : ∀ p,p.1≠output.info.storage ∨ r.out.isSome=Bool.true → memory p=caller p
  /-- Ordinary contiguous Tensor readback law for the allocated or supplied
  output. This is a layout fact, with no arithmetic values or result memory. -/
  outputIndex : ∀ i,i<r.shape.M*r.shape.N → output.physicalIndex i=offset (r.out.map TensorView.info)+i

def OutputViewBinding (r : GemmRequest) (t : TensorView) : Prop :=
  ∀ cell : MatrixCell r.shape,logicalAddress t (cell.1.val*r.shape.N+cell.2.val)=outputAddress r.shape (r.buffers t.info.storage) cell
theorem GemmAllocation.binding (alloc : GemmAllocation r caller live) : OutputViewBinding r alloc.output := by
  intro cell
  have hi : cell.1.val*r.shape.N+cell.2.val<r.shape.M*r.shape.N := by
    have hm:=cell.1.isLt;have hn:=cell.2.isLt;nlinarith
  simp only [logicalAddress,alloc.outputIndex _ hi,outputAddress,GemmRequest.buffers,gemmBuffers,Nat.add_assoc]

theorem GemmAllocation.separate (r : GemmRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : GemmAllocation r caller live) : (r.buffers alloc.output.info.storage).Separate := by
  have sep : ∀ t∈r.views,t.info.storage≠alloc.output.info.storage := by
    intro t ht
    cases eq : r.out with
    | none=>intro h;have live:=inputs t ht;rw [h] at live;exact alloc.fresh eq live
    | some output=>
      have identity : alloc.output=output:=by simpa [GemmDescriptor,eq] using alloc.descriptor
      have out:=r.contract.output output.info (by simp [eq])
      have member : t.info∈Direct.inputs r.a.info (some r.aScale.info) r.b.info r.bScale.info Bool.false := by
        simpa [GemmRequest.views,Direct.inputs] using List.mem_map.mpr ⟨t,ht,rfl⟩
      simpa [identity] using (out.separated t.info member).symm
  exact ⟨sep _ (by simp [GemmRequest.views]),sep _ (by simp [GemmRequest.views]),
    sep _ (by simp [GemmRequest.views]),sep _ (by simp [GemmRequest.views])⟩

structure HostResult (F H : Type) where
  tensor : TensorView
  memory : Memory F H
  currentDevice : Nat
def optionalArgument (out : Option TensorView) : OptionalArgument :=
  match out with | none=>.none | some t=>.tensor t.info
def checkGemmRequest (r : GemmRequest) :=
  checkGemm (.tensor r.a.info) (.tensor r.aScale.info) (.tensor r.b.info) (.tensor r.bScale.info) (optionalArgument r.out)
theorem checkGemmRequest_ok (r : GemmRequest) : checkGemmRequest r=.ok () := by
  apply (checkGemm_iff _ _ _ _ _).mpr
  cases eq : r.out <;> simpa [optionalArgument,eq] using r.contract

def runGemm (r : GemmRequest) (a : Ops I F H) (callerDevice : Nat)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (caller : Memory F H) (live : Nat → Prop) (alloc : GemmAllocation r caller live) : Except InputError (HostResult F H) := do
  checkGemmRequest r
  let result:=wrapper a r.shape (r.buffers alloc.output.info.storage) r.owner callerDevice trials chosen alloc.memory
  return ⟨alloc.output,result.memory,result.currentDevice⟩

theorem gemm_complete (r : GemmRequest) (a : Ops I F H) (callerDevice : Nat)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (alloc : GemmAllocation r caller live) :
    ∃ result,runGemm r a callerDevice trials chosen caller live alloc=.ok result ∧
      GemmDescriptor r result.tensor ∧ OutputViewBinding r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (outputAddress r.shape (r.buffers result.tensor.info.storage) cell)=
        .inr (wordReference a r.shape chosen.config (r.buffers result.tensor.info.storage) chosen.selected chosen.schedule caller cell)) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p → result.memory p=caller p) := by
  let done:=wrapper a r.shape (r.buffers alloc.output.info.storage) r.owner callerDevice trials chosen alloc.memory
  refine ⟨⟨alloc.output,done.memory,done.currentDevice⟩,?_,alloc.descriptor,alloc.binding,?_,?_,?_⟩
  · simp [runGemm,checkGemmRequest_ok,Except.bind,Except.pure,bind,pure,done]
  · exact (wrapper_returns _ _ _ _ _ _ _ _).2
  · intro cell
    change done.memory _=_
    rw [wrapper_word a _ _ (alloc.separate r caller live inputs)]
    apply congrArg Sum.inr
    apply reference_current a _ _ _ (alloc.separate r caller live inputs)
    intro p hp
    exact alloc.frame p (Or.inl hp)
  · intro p hp outside
    change done.memory p=caller p
    rw [wrapper_full_frame _ _ _ _ _ _ _ _ _ outside]
    apply alloc.frame
    cases eq : r.out with
    | none=>left;intro h;rw [h] at hp;exact alloc.fresh eq hp
    | some t=>right;simp [eq]

#axiomsClean original_pointer
#axiomsClean GemmAllocation.separate
#axiomsClean checkGemmRequest_ok
#axiomsClean gemm_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
