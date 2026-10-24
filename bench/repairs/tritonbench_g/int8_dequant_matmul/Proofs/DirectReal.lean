import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectPointers
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ideal
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
noncomputable section
set_option maxHeartbeats 1200000

/-- Independent decoded mathematical input from original tensor storage.
Only signed INT8 inputs are decoded; real scales/bias remain arbitrary reals. -/
def logicalInput (r : Request) (m : Memory ℝ ℝ) : RealInput (requestShape r) where
  A := fun row k=>((read8 m (logicalAddress r.a (row*(requestShape r).K+k))).toInt : ℝ)
  B := fun k n=>((read8 m (logicalAddress r.b (k*(requestShape r).N+n))).toInt : ℝ)
  x := fun row=>read32 realOps m (logicalAddress r.x row)
  w := fun n=>read32 realOps m (logicalAddress r.w n)
  bias := r.bias.map (fun t n=>read32 realOps m (logicalAddress t n))

theorem target_original (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (m : Memory ℝ ℝ) (out : Nat) (cell : MatrixCell (requestShape r)) :
    target (requestShape r) (idealInput (requestShape r) (requestBuffers r out) (hasBias r) m) cell.1 cell.2=
      target (requestShape r) (logicalInput r m) cell.1 cell.2 := by
  have hr:=cell.1.isLt
  have hn:=cell.2.isLt
  have hx : cell.1.val<r.x.info.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.1,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
  have hw : cell.2.val<r.w.info.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.2,List.prod_cons,List.prod_nil,Nat.mul_one] using hn
  have sx:=original_read 4 base r.x laws.x r.contract.contiguous.2.2.1 m cell.1.val hx
  have sw:=original_read 4 base r.w laws.w r.contract.contiguous.2.2.2 m cell.2.val hw
  have dot : rsum (fun k=>(idealInput (requestShape r) (requestBuffers r out) (hasBias r) m).A cell.1 k*
      (idealInput (requestShape r) (requestBuffers r out) (hasBias r) m).B k cell.2) (requestShape r).K=
      rsum (fun k=>(logicalInput r m).A cell.1 k*(logicalInput r m).B k cell.2) (requestShape r).K := by
    apply rsum_congr
    intro k hk
    obtain ⟨ha,hb,_,_⟩:=original_operands base r laws m out cell.1 cell.2 k hr hn hk
    simp only [idealInput,logicalInput,read8]
    rw [ha,hb]
  have row : (idealInput (requestShape r) (requestBuffers r out) (hasBias r) m).x cell.1=(logicalInput r m).x cell.1 := by
    change read32 realOps m (r.x.info.storage,r.x.info.offset+cell.1.val)=read32 realOps m (logicalAddress r.x cell.1.val)
    simp only [read32,sx]
  have col : (idealInput (requestShape r) (requestBuffers r out) (hasBias r) m).w cell.2=(logicalInput r m).w cell.2 := by
    change read32 realOps m (r.w.info.storage,r.w.info.offset+cell.2.val)=read32 realOps m (logicalAddress r.w cell.2.val)
    simp only [read32,sw]
  have hb : biasAt (idealInput (requestShape r) (requestBuffers r out) (hasBias r) m) cell.2=
      biasAt (logicalInput r m) cell.2 := by
    cases eq : r.bias with
    | none=>simp [biasAt,idealInput,logicalInput,hasBias,eq]
    | some t=>
      have tc:=r.contract.biasContract t.info (by simp [eq])
      have nt : cell.2.val<t.info.metadata.numel := by
        simpa only [VeriTile.Torch.TensorMetadata.numel,tc.descriptor.2.2.1,List.prod_cons,List.prod_nil,Nat.mul_one] using hn
      have read:=original_read 4 base t (laws.bias t eq) tc.direct.1 m cell.2.val nt
      simp only [biasAt,idealInput,logicalInput,hasBias,eq,Option.isSome_some,↓reduceIte,Option.map_some,Option.getD_some]
      simp only [requestBuffers,buffers,biasOffset,eq,Option.map_some,Option.getD_some]
      simp only [read32,read]
  simp only [target,dot,row,col,hb]

theorem real_complete (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (callerDevice : Nat) (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputs : InputsLive r live) (alloc : Allocation r caller live) :
    ∃ result, run r realOps (1/16129) callerDevice benchmarks chosen caller live alloc=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),read16 realOps result.memory (outputAddress (requestShape r) (requestBuffers r alloc.storage) cell)=
        target (requestShape r) (logicalInput r caller) cell.1 cell.2) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  obtain ⟨result,ok,tensor,device,values,frame⟩:=run_complete r realOps (1/16129) callerDevice benchmarks chosen caller live inputs alloc
  refine ⟨result,ok,tensor,device,?_,frame⟩
  intro cell
  rw [read16,values cell]
  rw [real_reference,target_original base r laws caller alloc.storage cell]

#axiomsClean target_original
#axiomsClean real_complete
#specNonCircular target avoiding [run,wrapper,invoke,partialValue,wordReference]
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
