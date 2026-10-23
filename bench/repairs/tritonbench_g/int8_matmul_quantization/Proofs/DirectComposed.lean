import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectComposedData
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectWords

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 2500000

def ComposedRequest.shape (r : ComposedRequest) := Direct.shape r.fpa.info r.b.info
structure ComposedLaws (base : Nat → Nat) (r : ComposedRequest) : Prop where
  fpa : TensorLaws base r.fpa
  b : TensorLaws base r.b
  bScale : TensorLaws base r.bScale
  output : ∀ t,r.out=some t → TensorLaws base t
def extendedLive (live : Nat → Prop) (data scales : Nat) (storage : Nat) : Prop :=
  live storage ∨ storage=data ∨ storage=scales
def producerMemory (r : RowRequest) (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop)
    (alloc : RowAllocation r caller live) (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K) :=
  quantWrapper a (r.buffers alloc.data.info.storage alloc.scales.info.storage) trials chosen alloc.memory
theorem producer_values (r : RowRequest) (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K) (idx : QuantIndex r.G r.K) :
    producerMemory r a caller live alloc trials chosen ((r.buffers alloc.data.info.storage alloc.scales.info.storage).writeAddress idx)=
      quantExpected a (r.buffers alloc.data.info.storage alloc.scales.info.storage) chosen.plan caller idx := by
  let q:=r.buffers alloc.data.info.storage alloc.scales.info.storage
  have sep : q.Separate:=alloc.separate r caller live input
  have same : QuantInputsAgree q alloc.memory caller := by
    intro p hp;apply alloc.frame
    · simpa only [hp] using sep.1
    · simpa only [hp] using sep.2.1
  unfold producerMemory
  rw [quantWrapper_at a _ sep]
  cases idx with
  | inl cell=>exact congrArg Sum.inl (quantValue_current a q _ _ _ same cell)
  | inr g=>simp only [quantExpected,quantScale_current a q _ _ _ same];rfl
theorem producer_frame (r : RowRequest) (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K) (p : Address) (owned : live p.1) :
    producerMemory r a caller live alloc trials chosen p=caller p := by
  have hd : p.1≠alloc.data.info.storage:=by intro eq;rw [eq] at owned;exact alloc.dataFresh owned
  have hs : p.1≠alloc.scales.info.storage:=by intro eq;rw [eq] at owned;exact alloc.scaleFresh owned
  unfold producerMemory
  rw [quantWrapper_frame a _ (alloc.separate r caller live input)]
  · exact alloc.frame p hd hs
  · intro idx eq;have first:=congrArg Prod.fst eq
    cases idx with
    | inl cell=>exact hd first
    | inr g=>exact hs first

theorem produced_link (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : RowAllocation r.row caller live) (out : Nat) :
    CompositionLink (r.row.buffers alloc.data.info.storage alloc.scales.info.storage)
      ((producedRequest r caller live inputs alloc).buffers out) := ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩
theorem produced_live (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : RowAllocation r.row caller live) :
    (producedRequest r caller live inputs alloc).live (extendedLive live alloc.data.info.storage alloc.scales.info.storage) := by
  intro t ht
  change t∈[composedA r alloc.data,composedScale r alloc.scales,r.b,r.bScale] at ht
  simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with eq|eq|eq|eq <;> subst t
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr rfl)
  · exact Or.inl (inputs r.b (by simp [ComposedRequest.views]))
  · exact Or.inl (inputs r.bScale (by simp [ComposedRequest.views]))

def checkComposedRequest (r : ComposedRequest) :=
  checkComposed (.tensor r.fpa.info) (.tensor r.b.info) (.tensor r.bScale.info) (optionalArgument r.out)
theorem checkComposedRequest_ok (r : ComposedRequest) : checkComposedRequest r=.ok () := by
  apply (checkComposed_iff _ _ _ _).mpr
  cases eq : r.out <;> simpa [optionalArgument,eq] using r.contract

/-- Actual producer memory is the caller memory for the second allocation and
GEMM. Repeated admission of the produced tensors is derived, never assumed. -/
def runComposed (r : ComposedRequest) (a : Ops I F H) (callerDevice : Nat)
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (output : GemmAllocation (producedRequest r caller live inputs producer)
      (producerMemory r.row a caller live producer qtrials qchosen) (extendedLive live producer.data.info.storage producer.scales.info.storage)) :
    Except InputError (HostResult F H) := do
  checkComposedRequest r
  checkRow (.tensor r.fpa.info)
  runGemm (producedRequest r caller live inputs producer) a callerDevice trials chosen
    (producerMemory r.row a caller live producer qtrials qchosen) _ output

theorem composed_complete (r : ComposedRequest) (a : Ops I F H) (callerDevice : Nat)
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (output : GemmAllocation (producedRequest r caller live inputs producer)
      (producerMemory r.row a caller live producer qtrials qchosen) (extendedLive live producer.data.info.storage producer.scales.info.storage)) :
    ∃ result,runComposed r a callerDevice caller live inputs producer qtrials qchosen trials chosen output=.ok result ∧
      GemmDescriptor (producedRequest r caller live inputs producer) result.tensor ∧ OutputViewBinding (producedRequest r caller live inputs producer) result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (outputAddress r.shape ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage) cell)=
        .inr (wordReference a r.shape chosen.config ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage) chosen.selected chosen.schedule
          (producerMemory r.row a caller live producer qtrials qchosen) cell)) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage) p → result.memory p=caller p) := by
  obtain ⟨result,ok,descriptor,readback,device,values,frame⟩:=gemm_complete (producedRequest r caller live inputs producer) a callerDevice trials chosen
    (producerMemory r.row a caller live producer qtrials qchosen) _ (produced_live r caller live inputs producer) output
  refine ⟨result,?_,descriptor,readback,device,values,?_⟩
  · simp [runComposed,checkComposedRequest_ok,(checkRow_iff (.tensor r.fpa.info)).mpr r.contract.1,Except.bind,bind,ok]
  · intro p owned outside
    rw [frame p (Or.inl owned) outside]
    exact producer_frame r.row a caller live (r.fpaLive live inputs) producer qtrials qchosen p owned

/-- Reads from the original B descriptor in either admitted physical layout. -/
theorem composed_b_address (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (out k n : Nat) (hk : k<r.shape.K) (hn : n<r.shape.N) :
    ((producedRequest r caller live inputs producer).buffers out).bIndex r.shape k n=r.b.physicalIndex (k*r.shape.N+n) := by
  have bound : k*r.shape.N+n<r.b.info.metadata.numel := by
    rw [matrix_numel _ r.contract.2.bTensor.rank,r.contract.2.shape.1]
    change k*cols r.b.info+n<cols r.fpa.info*cols r.b.info
    change k<cols r.fpa.info at hk;change n<cols r.b.info at hn;nlinarith
  cases hc : r.b.info.metadata.isContiguous with
  | «true»=>
    rw [laws.b.contiguous hc _ bound]
    simp [producedRequest,GemmRequest.buffers,gemmBuffers,Buffers.bIndex,columnMajor,hc]
  | «false»=>
    have trans : r.b.info.transposeContiguous=Bool.true:=by
      rcases r.contract.2.bTensor.dense with h|h
      · simp [hc] at h
      · exact h.2
    have hi : k<rows r.b.info:=by rw [r.contract.2.shape.1];exact hk
    rw [show r.shape.N=cols r.b.info from rfl,laws.b.column trans k n hi hn,r.contract.2.shape.1]
    simp [producedRequest,GemmRequest.buffers,gemmBuffers,Buffers.bIndex,columnMajor,hc,ComposedRequest.shape,shape,Nat.add_assoc]

def composedRealInput (r : ComposedRequest) (m : Memory ℝ ℝ) : RealInput r.shape where
  A := fun row k=>((rowValueTarget r.row m row k).toInt : ℝ)
  B := fun k n=>((read8 m (logicalAddress r.b (k*r.shape.N+n))).toInt : ℝ)
  aScale := rowScaleTarget r.row m
  bScale := fun n=>read32 realOps.toOps m (logicalAddress r.bScale n)

theorem composed_a_value (r : ComposedRequest) (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (out : Nat) (row : Fin r.shape.M) (k : Fin r.shape.K) :
    read8 (producerMemory r.row a caller live producer qtrials qchosen)
      (((producedRequest r caller live inputs producer).buffers out).a,
        ((producedRequest r caller live inputs producer).buffers out).aOffset+row.val*r.shape.K+k.val)=
      quantValue a (r.row.buffers producer.data.info.storage producer.scales.info.storage) qchosen.plan caller (row,k) := by
  have h:=producer_values r.row a caller live (r.fpaLive live inputs) producer qtrials qchosen (.inl (row,k))
  simp only [QuantBuffers.writeAddress,quantExpected] at h
  have addr : (((producedRequest r caller live inputs producer).buffers out).a,
        ((producedRequest r caller live inputs producer).buffers out).aOffset+row.val*r.shape.K+k.val)=
      (r.row.buffers producer.data.info.storage producer.scales.info.storage).dataAddress (row,k) := by
    simp [producedRequest,GemmRequest.buffers,gemmBuffers,composedA,canonicalAllocated,
      RowRequest.buffers,rowBuffers,QuantBuffers.dataAddress,RowRequest.K,ComposedRequest.shape,ComposedRequest.row,shape]
  rw [addr]
  exact congrArg (fun value : Cell F H=>match value with | .inl byte=>byte | _=>0) h

theorem composed_b_value (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (out : Nat) (k : Fin r.shape.K) (n : Fin r.shape.N) :
    read8 (producerMemory r.row a caller live producer qtrials qchosen)
      (((producedRequest r caller live inputs producer).buffers out).b,
        ((producedRequest r caller live inputs producer).buffers out).bIndex r.shape k n)=
      read8 caller (logicalAddress r.b (k.val*r.shape.N+n.val)) := by
  rw [composed_b_address base r laws caller live inputs producer out k n k.isLt n.isLt]
  change read8 (producerMemory r.row a caller live producer qtrials qchosen) (logicalAddress r.b _)=_
  unfold read8 Int8DequantMatmul.read8
  rw [producer_frame r.row a caller live (r.fpaLive live inputs) producer qtrials qchosen]
  exact inputs r.b (by simp [ComposedRequest.views])

theorem composed_scale_values (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (out : Nat) (cell : MatrixCell r.shape) :
    loadScale a ((producedRequest r caller live inputs producer).buffers out).aScalePrecision
      (producerMemory r.row a caller live producer qtrials qchosen)
      (((producedRequest r caller live inputs producer).buffers out).aScale,
        ((producedRequest r caller live inputs producer).buffers out).aScaleOffset+cell.1.val)=
      a.widen (a.narrow (quantScale a (r.row.buffers producer.data.info.storage producer.scales.info.storage) qchosen.plan caller cell.1)) ∧
    loadScale a ((producedRequest r caller live inputs producer).buffers out).bScalePrecision
      (producerMemory r.row a caller live producer qtrials qchosen)
      (((producedRequest r caller live inputs producer).buffers out).bScale,
        ((producedRequest r caller live inputs producer).buffers out).bScaleOffset+cell.2.val)=
      read32 a.toOps caller (logicalAddress r.bScale cell.2.val) := by
  constructor
  · have h:=producer_values r.row a caller live (r.fpaLive live inputs) producer qtrials qchosen (.inr cell.1)
    simp only [QuantBuffers.writeAddress,quantExpected] at h
    have addr : (((producedRequest r caller live inputs producer).buffers out).aScale,
        ((producedRequest r caller live inputs producer).buffers out).aScaleOffset+cell.1.val)=
        (r.row.buffers producer.data.info.storage producer.scales.info.storage).scaleAddress cell.1 := rfl
    rw [addr]
    change a.widen (read16 a.toOps _ _) = _
    change producerMemory r.row a caller live producer qtrials qchosen
      ((r.row.buffers producer.data.info.storage producer.scales.info.storage).scaleAddress cell.1)=
      .inr (.inr (a.narrow (quantScale a (r.row.buffers producer.data.info.storage producer.scales.info.storage) qchosen.plan caller cell.1))) at h
    exact congrArg (fun value : Cell F H=>a.widen (match value with | .inr (.inr v)=>v | _=>a.hzero)) h
  · have hi : cell.2.val<r.bScale.info.metadata.numel:=by
      simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.2.shape.2,List.prod_cons,List.prod_nil,Nat.mul_one,ComposedRequest.shape,shape] using cell.2.isLt
    have addr : (((producedRequest r caller live inputs producer).buffers out).bScale,
        ((producedRequest r caller live inputs producer).buffers out).bScaleOffset+cell.2.val)=logicalAddress r.bScale cell.2.val := by
      simp [producedRequest,GemmRequest.buffers,gemmBuffers,logicalAddress,laws.bScale.contiguous (row_contiguous r.contract.2.bScaleTensor) _ hi]
    rw [addr]
    change read32 a.toOps (producerMemory r.row a caller live producer qtrials qchosen) (logicalAddress r.bScale cell.2.val)=_
    unfold read32 Int8DequantMatmul.read32
    rw [producer_frame r.row a caller live (r.fpaLive live inputs) producer qtrials qchosen]
    exact inputs r.bScale (by simp [ComposedRequest.views])

theorem composed_target_original (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (out : Nat) (cell : MatrixCell r.shape) :
    target r.shape (realInput r.shape ((producedRequest r caller live inputs producer).buffers out)
      (producerMemory r.row realOps caller live producer qtrials qchosen)) cell.1 cell.2=
      target r.shape (composedRealInput r caller) cell.1 cell.2 := by
  have dot : Int8DequantMatmul.rsum (fun k=>(realInput r.shape ((producedRequest r caller live inputs producer).buffers out)
      (producerMemory r.row realOps caller live producer qtrials qchosen)).A cell.1 k*
      (realInput r.shape ((producedRequest r caller live inputs producer).buffers out)
      (producerMemory r.row realOps caller live producer qtrials qchosen)).B k cell.2) r.shape.K=
      Int8DequantMatmul.rsum (fun k=>(composedRealInput r caller).A cell.1 k*(composedRealInput r caller).B k cell.2) r.shape.K := by
    apply Int8DequantMatmul.rsum_congr
    intro k hk
    have av:=composed_a_value r realOps caller live inputs producer qtrials qchosen out cell.1 ⟨k,hk⟩
    rw [row_value_original base r.row laws.fpa] at av
    have bv:=composed_b_value base r laws realOps caller live inputs producer qtrials qchosen out ⟨k,hk⟩ cell.2
    simp only [realInput,composedRealInput,av,bv]
  obtain ⟨asv,bsv⟩:=composed_scale_values base r laws realOps caller live inputs producer qtrials qchosen out cell
  rw [row_scale_original base r.row laws.fpa] at asv
  exact congrArg₂ (fun (x y : ℝ)=>x*y) (congrArg₂ (fun (x y : ℝ)=>x*y) dot asv) bsv

theorem composed_real_complete (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r) (callerDevice : Nat)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (output : GemmAllocation (producedRequest r caller live inputs producer)
      (producerMemory r.row realOps caller live producer qtrials qchosen) (extendedLive live producer.data.info.storage producer.scales.info.storage)) :
    ∃ result,runComposed r realOps callerDevice caller live inputs producer qtrials qchosen trials chosen output=.ok result ∧
      GemmDescriptor (producedRequest r caller live inputs producer) result.tensor ∧ OutputViewBinding (producedRequest r caller live inputs producer) result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (outputAddress r.shape ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage) cell)=
        .inr (embedReal ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage).outPrecision
          (target r.shape (composedRealInput r caller) cell.1 cell.2))) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape ((producedRequest r caller live inputs producer).buffers result.tensor.info.storage) p → result.memory p=caller p) := by
  obtain ⟨result,ok,descriptor,readback,device,values,frame⟩:=composed_complete r realOps callerDevice caller live inputs producer qtrials qchosen trials chosen output
  refine ⟨result,ok,descriptor,readback,device,?_,frame⟩
  intro cell
  rw [values cell,real_reference,composed_target_original base r laws]

#axiomsClean produced_link
#axiomsClean produced_live
#axiomsClean checkComposedRequest_ok
#axiomsClean composed_complete
#axiomsClean composed_b_address
#axiomsClean composed_target_original
#axiomsClean composed_real_complete
#specNonCircular composedRealInput avoiding [producerMemory,runComposed,composed,quantWrapper,wordReference]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
