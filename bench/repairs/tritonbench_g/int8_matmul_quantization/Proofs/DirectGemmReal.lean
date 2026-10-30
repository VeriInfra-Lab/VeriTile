import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectData
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.QuantReference

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

theorem GemmRequest.aScale_contract (r : GemmRequest) :
    TensorContract r.aScale.info [.other "float16",.floating .fp32] 1 Bool.false ∧
      r.aScale.info.metadata.shape=[r.shape.M] := by
  obtain ⟨t,eq,h,shape⟩:=r.contract.aScale rfl
  have eq':r.aScale.info=t:=Option.some.inj eq
  simpa only [←eq'] using And.intro h shape

theorem gemm_a_address (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (out row k : Nat) (hr : row<r.shape.M) (hk : k<r.shape.K) :
    ((r.buffers out).a,(r.buffers out).aOffset+row*r.shape.K+k)=logicalAddress r.a (row*r.shape.K+k) := by
  have bound : row*r.shape.K+k<r.a.info.metadata.numel := by
    rw [matrix_numel _ r.contract.aTensor.rank]
    change row*cols r.a.info+k<rows r.a.info*cols r.a.info
    change row<rows r.a.info at hr;change k<cols r.a.info at hk
    nlinarith
  have law:=laws.inputs r.a (by simp [GemmRequest.views])
  simp [GemmRequest.buffers,gemmBuffers,logicalAddress,law.contiguous (row_contiguous r.contract.aTensor) _ bound,Nat.add_assoc]

theorem gemm_b_address (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (out k n : Nat) (hk : k<r.shape.K) (hn : n<r.shape.N) :
    ((r.buffers out).b,(r.buffers out).bIndex r.shape k n)=logicalAddress r.b (k*r.shape.N+n) := by
  have bound : k*r.shape.N+n<r.b.info.metadata.numel := by
    rw [matrix_numel _ r.contract.bTensor.rank,r.contract.shape.1]
    change k*cols r.b.info+n<cols r.a.info*cols r.b.info
    change k<cols r.a.info at hk;change n<cols r.b.info at hn
    nlinarith
  have law:=laws.inputs r.b (by simp [GemmRequest.views])
  cases hc : r.b.info.metadata.isContiguous with
  | «true»=>simp [GemmRequest.buffers,gemmBuffers,Buffers.bIndex,columnMajor,hc,logicalAddress,law.contiguous hc _ bound]
  | «false»=>
    have trans : r.b.info.transposeContiguous=Bool.true := by
      rcases r.contract.bTensor.dense with h|h
      · simp [hc] at h
      · exact h.2
    have hi : k<rows r.b.info := by rw [r.contract.shape.1];exact hk
    have h:=law.column trans k n hi hn
    simp only [r.contract.shape.1] at h
    simpa [GemmRequest.buffers,gemmBuffers,Buffers.bIndex,columnMajor,hc,logicalAddress,GemmRequest.shape,shape,Nat.add_assoc] using congrArg (fun idx=>(r.b.info.storage,idx)) h.symm

theorem gemm_scale_addresses (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (out : Nat) (cell : MatrixCell r.shape) :
    ((r.buffers out).aScale,(r.buffers out).aScaleOffset+cell.1.val)=logicalAddress r.aScale cell.1.val ∧
    ((r.buffers out).bScale,(r.buffers out).bScaleOffset+cell.2.val)=logicalAddress r.bScale cell.2.val := by
  have ha : cell.1.val<r.aScale.info.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,r.aScale_contract.2] using cell.1.isLt
  have hb : cell.2.val<r.bScale.info.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2,List.prod_cons,List.prod_nil,Nat.mul_one,GemmRequest.shape,shape] using cell.2.isLt
  have al:=laws.inputs r.aScale (by simp [GemmRequest.views])
  have bl:=laws.inputs r.bScale (by simp [GemmRequest.views])
  constructor
  · simp [GemmRequest.buffers,gemmBuffers,offset,logicalAddress,al.contiguous (row_contiguous r.aScale_contract.1) _ ha]
  · simp [GemmRequest.buffers,gemmBuffers,logicalAddress,bl.contiguous (row_contiguous r.contract.bScaleTensor) _ hb]

def originalRealInput (r : GemmRequest) (m : Memory ℝ ℝ) : RealInput r.shape where
  A := fun row k=>((read8 m (logicalAddress r.a (row*r.shape.K+k))).toInt : ℝ)
  B := fun k n=>((read8 m (logicalAddress r.b (k*r.shape.N+n))).toInt : ℝ)
  aScale := fun row=>loadScale realOps (scalePrecision (some r.aScale.info) Bool.false) m (logicalAddress r.aScale row)
  bScale := fun n=>loadScale realOps .single m (logicalAddress r.bScale n)

theorem gemm_target_original (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (m : Memory ℝ ℝ) (out : Nat) (cell : MatrixCell r.shape) :
    target r.shape (realInput r.shape (r.buffers out) m) cell.1 cell.2=
      target r.shape (originalRealInput r m) cell.1 cell.2 := by
  have dot : Int8DequantMatmul.rsum (fun k=>(realInput r.shape (r.buffers out) m).A cell.1 k*(realInput r.shape (r.buffers out) m).B k cell.2) r.shape.K=
      Int8DequantMatmul.rsum (fun k=>(originalRealInput r m).A cell.1 k*(originalRealInput r m).B k cell.2) r.shape.K := by
    apply Int8DequantMatmul.rsum_congr
    intro k hk
    simp only [realInput,originalRealInput,gemm_a_address base r laws out _ k cell.1.isLt hk,
      gemm_b_address base r laws out k _ hk cell.2.isLt]
  obtain ⟨ha,hb⟩:=gemm_scale_addresses base r laws out cell
  simp only [target,dot]
  simp only [realInput,originalRealInput,ha,hb]
  rfl

theorem gemm_real_complete (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (callerDevice : Nat) (trials : List (Invocation r.shape)) (chosen : Invocation r.shape)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : r.live live) (alloc : GemmAllocation r caller live) :
    ∃ result,runGemm r realOps callerDevice trials chosen caller live alloc=.ok result ∧
      GemmDescriptor r result.tensor ∧ OutputViewBinding r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (outputAddress r.shape (r.buffers result.tensor.info.storage) cell)=
        .inr (embedReal (r.buffers result.tensor.info.storage).outPrecision (target r.shape (originalRealInput r caller) cell.1 cell.2))) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p → result.memory p=caller p) := by
  obtain ⟨result,ok,descriptor,readback,device,values,frame⟩:=gemm_complete r realOps callerDevice trials chosen caller live inputs alloc
  refine ⟨result,ok,descriptor,readback,device,?_,frame⟩
  intro cell
  rw [values cell,real_reference,gemm_target_original base r laws]

#axiomsClean gemm_a_address
#axiomsClean gemm_b_address
#axiomsClean gemm_scale_addresses
#axiomsClean gemm_target_original
#axiomsClean gemm_real_complete
#specNonCircular originalRealInput avoiding [wrapper,runGemm,wordReference]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
