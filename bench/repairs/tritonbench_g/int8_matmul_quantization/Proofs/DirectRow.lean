import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectData
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.QuantReference

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

structure RowRequest where
  input : TensorView
  contract : RowContract input.info
def RowRequest.G (r : RowRequest) := rows r.input.info
def RowRequest.K (r : RowRequest) := cols r.input.info
def RowRequest.buffers (r : RowRequest) (data scales : Nat) := rowBuffers r.input.info data scales

structure RowAllocation (r : RowRequest) (caller : Memory F H) (live : Nat → Prop) where
  data : TensorView
  scales : TensorView
  memory : Memory F H
  dataDescriptor : FreshDescriptor data (.other "int8") [r.G,r.K] [max r.K 1,1] r.input.info.metadata.device
  scaleDescriptor : FreshDescriptor scales (.other "float16") [r.G] [1] r.input.info.metadata.device
  scaleAlignment : 0<scales.info.metadata.numel → scales.info.metadata.dataPtr.toNat%2=0
  dataFresh : ¬live data.info.storage
  scaleFresh : ¬live scales.info.storage
  distinct : data.info.storage≠scales.info.storage
  frame : ∀ p,p.1≠data.info.storage → p.1≠scales.info.storage → memory p=caller p
  dataIndex : ∀ i,i<r.G*r.K → data.physicalIndex i=i
  scaleIndex : ∀ i,i<r.G → scales.physicalIndex i=i

def RowViewBinding (r : RowRequest) (data scales : TensorView) : Prop :=
  (∀ cell : QuantCell r.G r.K,logicalAddress data (cell.1.val*r.K+cell.2.val)=
    (r.buffers data.info.storage scales.info.storage).dataAddress cell) ∧
  (∀ g : Fin r.G,logicalAddress scales g.val=(r.buffers data.info.storage scales.info.storage).scaleAddress g)
theorem RowAllocation.binding (alloc : RowAllocation r caller live) : RowViewBinding r alloc.data alloc.scales := by
  constructor
  · intro cell
    have hi : cell.1.val*r.K+cell.2.val<r.G*r.K := by have hg:=cell.1.isLt;have hk:=cell.2.isLt;nlinarith
    unfold logicalAddress
    rw [alloc.dataIndex _ hi]
    simp [RowRequest.buffers,rowBuffers,QuantBuffers.dataAddress]
  · intro g
    simp [logicalAddress,alloc.scaleIndex _ g.isLt,RowRequest.buffers,rowBuffers,QuantBuffers.scaleAddress]
theorem RowAllocation.separate (r : RowRequest) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.input.info.storage) (alloc : RowAllocation r caller live) :
    (r.buffers alloc.data.info.storage alloc.scales.info.storage).Separate := by
  refine ⟨?_,?_,alloc.distinct⟩
  · intro h;change r.input.info.storage=alloc.data.info.storage at h;rw [h] at input;exact alloc.dataFresh input
  · intro h;change r.input.info.storage=alloc.scales.info.storage at h;rw [h] at input;exact alloc.scaleFresh input

structure RowResult (F H : Type) where
  data : TensorView
  scales : TensorView
  memory : Memory F H
  currentDevice : Nat
def runRow (r : RowRequest) (a : Ops I F H) (callerDevice : Nat)
    (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (alloc : RowAllocation r caller live) : Except InputError (RowResult F H) := do
  checkRow (.tensor r.input.info)
  return ⟨alloc.data,alloc.scales,quantWrapper a (r.buffers alloc.data.info.storage alloc.scales.info.storage) trials chosen alloc.memory,callerDevice⟩

theorem row_complete (r : RowRequest) (a : Ops I F H) (callerDevice : Nat)
    (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K)
    (caller : Memory F H) (live : Nat → Prop) (input : live r.input.info.storage) (alloc : RowAllocation r caller live) :
    ∃ result,runRow r a callerDevice trials chosen caller live alloc=.ok result ∧
      FreshDescriptor result.data (.other "int8") [r.G,r.K] [max r.K 1,1] r.input.info.metadata.device ∧
      FreshDescriptor result.scales (.other "float16") [r.G] [1] r.input.info.metadata.device ∧
      RowViewBinding r result.data result.scales ∧ result.currentDevice=callerDevice ∧
      (∀ idx : QuantIndex r.G r.K,result.memory ((r.buffers result.data.info.storage result.scales.info.storage).writeAddress idx)=
        quantExpected a (r.buffers result.data.info.storage result.scales.info.storage) chosen.plan caller idx) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  let q:=r.buffers alloc.data.info.storage alloc.scales.info.storage
  have sep:=alloc.separate r caller live input
  have same : QuantInputsAgree q alloc.memory caller := by
    intro p hp
    apply alloc.frame
    · simpa only [hp] using sep.1
    · simpa only [hp] using sep.2.1
  refine ⟨⟨alloc.data,alloc.scales,quantWrapper a q trials chosen alloc.memory,callerDevice⟩,?_,alloc.dataDescriptor,alloc.scaleDescriptor,alloc.binding,rfl,?_,?_⟩
  · simp [runRow,(checkRow_iff (.tensor r.input.info)).mpr r.contract,Except.bind,Except.pure,bind,pure,q]
  · intro idx
    change quantWrapper a q trials chosen alloc.memory (q.writeAddress idx)=quantExpected a q chosen.plan caller idx
    rw [quantWrapper_at a q sep]
    cases idx with
    | inl cell=>exact congrArg Sum.inl (quantValue_current a q chosen.plan alloc.memory caller same cell)
    | inr g=>simp only [quantExpected,quantScale_current a q chosen.plan alloc.memory caller same]
  · intro p hp
    have hd : p.1≠alloc.data.info.storage:=by intro h;rw [h] at hp;exact alloc.dataFresh hp
    have hs : p.1≠alloc.scales.info.storage:=by intro h;rw [h] at hp;exact alloc.scaleFresh hp
    change quantWrapper a q trials chosen alloc.memory p=caller p
    rw [quantWrapper_frame a q sep trials chosen alloc.memory p]
    · exact alloc.frame p hd hs
    · intro idx h
      have first:=congrArg Prod.fst h
      cases idx with
      | inl x=>exact hd first
      | inr g=>exact hs first

theorem row_input_address (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (data scales : Nat) (g k : Nat) (hg : g<r.G) (hk : k<r.K) :
    (r.buffers data scales).inputAddress g k=logicalAddress r.input (g*r.K+k) := by
  have bound : g*r.K+k<r.input.info.metadata.numel := by
    rw [matrix_numel _ r.contract.input.rank]
    change g*cols r.input.info+k<rows r.input.info*cols r.input.info
    change g<rows r.input.info at hg;change k<cols r.input.info at hk
    nlinarith
  have positive : 1≤cols r.input.info:=by change k<cols r.input.info at hk;omega
  unfold logicalAddress
  rw [laws.contiguous (row_contiguous r.contract.input) _ bound]
  simp [RowRequest.buffers,rowBuffers,QuantBuffers.inputAddress,RowRequest.K,max_eq_left positive,Nat.add_assoc]

def originalRow (a : Ops I F H) (r : RowRequest) (m : Memory F H) (g k : Nat) : F :=
  read32 a.toOps m (logicalAddress r.input (g*r.K+k))
def rowScaleTarget (r : RowRequest) (m : Memory ℝ ℝ) (g : Nat) := quantScaleTarget (originalRow realOps r m g) r.K
def rowValueTarget (r : RowRequest) (m : Memory ℝ ℝ) (g k : Nat) := quantValueTarget (originalRow realOps r m g) r.K k

theorem maximumAbs_congr (f g : Nat → ℝ) (n : Nat) (same : ∀ k,k<n → f k=g k) : maximumAbs f n=maximumAbs g n := by
  induction n with
  | zero=>rfl
  | succ n ih=>rw [maximumAbs,maximumAbs,ih (fun k hk=>same k (by omega)),same n (by omega)]
theorem row_scale_original (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory ℝ ℝ) (g : Fin r.G) :
    quantScale realOps (r.buffers data scales) plan m g=rowScaleTarget r m g := by
  rw [quantScale_real]
  unfold rowScaleTarget quantScaleTarget
  congr 1
  apply maximumAbs_congr
  intro k hk
  rw [row_input_address base r laws data scales g k g.isLt hk]
  rfl
theorem row_value_original (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory ℝ ℝ) (cell : QuantCell r.G r.K) :
    quantValue realOps (r.buffers data scales) plan m cell=rowValueTarget r m cell.1 cell.2 := by
  unfold quantValue
  rw [row_scale_original base r laws,row_input_address base r laws data scales cell.1 cell.2 cell.1.isLt cell.2.isLt]
  rfl

theorem row_real_complete (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (callerDevice : Nat) (trials : List (QuantInvocation r.G r.K)) (chosen : QuantInvocation r.G r.K)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (input : live r.input.info.storage) (alloc : RowAllocation r caller live) :
    ∃ result,runRow r realOps callerDevice trials chosen caller live alloc=.ok result ∧
      FreshDescriptor result.data (.other "int8") [r.G,r.K] [max r.K 1,1] r.input.info.metadata.device ∧
      FreshDescriptor result.scales (.other "float16") [r.G] [1] r.input.info.metadata.device ∧
      RowViewBinding r result.data result.scales ∧ result.currentDevice=callerDevice ∧
      (∀ cell : QuantCell r.G r.K,result.memory (result.data.info.storage,cell.1.val*r.K+cell.2.val)=
        .inl (rowValueTarget r caller cell.1 cell.2)) ∧
      (∀ g : Fin r.G,result.memory (result.scales.info.storage,g.val)=.inr (.inr (rowScaleTarget r caller g))) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  obtain ⟨result,ok,data,scales,readback,device,values,frame⟩:=row_complete r realOps callerDevice trials chosen caller live input alloc
  refine ⟨result,ok,data,scales,readback,device,?_,?_,frame⟩
  · intro cell
    have h:=values (.inl cell)
    simp only [QuantBuffers.writeAddress,quantExpected] at h
    rw [row_value_original base r laws] at h
    simpa [RowRequest.buffers,rowBuffers,QuantBuffers.dataAddress,RowRequest.K] using h
  · intro g
    have h:=values (.inr g)
    simp only [QuantBuffers.writeAddress,quantExpected] at h
    rw [row_scale_original base r laws] at h
    simpa [RowRequest.buffers,rowBuffers,QuantBuffers.scaleAddress,realOps,Int8DequantMatmul.realOps] using h


#axiomsClean row_complete
#axiomsClean row_input_address
#axiomsClean row_real_complete
#specNonCircular rowScaleTarget avoiding [quantRun,quantWrapper,quantScale,Reduction.eval]
#specNonCircular rowValueTarget avoiding [quantRun,quantWrapper,quantValue,Reduction.eval]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
