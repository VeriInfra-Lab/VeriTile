import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectRow

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 3000000

structure ComposedRequest where
  fpa : TensorView
  b : TensorView
  bScale : TensorView
  out : Option TensorView
  contract : ComposedContract fpa.info b.info bScale.info (out.map TensorView.info)
def ComposedRequest.row (r : ComposedRequest) : RowRequest := ⟨r.fpa,r.contract.1⟩
def ComposedRequest.views (r : ComposedRequest) := [r.fpa,r.b,r.bScale]++r.out.toList
def ComposedRequest.live (r : ComposedRequest) (live : Nat → Prop) : Prop := ∀ t∈r.views,live t.info.storage
theorem ComposedRequest.fpaLive (r : ComposedRequest) (live : Nat → Prop) (h : r.live live) : live r.fpa.info.storage :=
  h r.fpa (by simp [ComposedRequest.views])

/-- Rechecking the actual produced INT8 tensor and FP16 scale succeeds from
the pre-allocation composed guards and ordinary fresh allocation facts. -/
theorem produced_admission (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : RowAllocation r.row caller live) :
    GemmContract alloc.data.info (some alloc.scales.info) r.b.info r.bScale.info (r.out.map TensorView.info) Bool.false := by
  have h:=r.contract.2
  have dm:=alloc.dataDescriptor.1
  have sm:=alloc.scaleDescriptor.1
  have dshape:=congrArg VeriTile.Torch.TensorMetadata.shape dm
  have dtype:=congrArg VeriTile.Torch.TensorMetadata.dtype dm
  have dlayout:=congrArg VeriTile.Torch.TensorMetadata.layout dm
  have ddevice:=congrArg VeriTile.Torch.TensorMetadata.device dm
  have dneg:=congrArg VeriTile.Torch.TensorMetadata.isNeg dm
  have dcontig:=congrArg VeriTile.Torch.TensorMetadata.isContiguous dm
  have sshape:=congrArg VeriTile.Torch.TensorMetadata.shape sm
  have stype:=congrArg VeriTile.Torch.TensorMetadata.dtype sm
  have slayout:=congrArg VeriTile.Torch.TensorMetadata.layout sm
  have sdevice:=congrArg VeriTile.Torch.TensorMetadata.device sm
  have sneg:=congrArg VeriTile.Torch.TensorMetadata.isNeg sm
  have scontig:=congrArg VeriTile.Torch.TensorMetadata.isContiguous sm
  dsimp only at dshape dtype dlayout ddevice dneg dcontig sshape stype slayout sdevice sneg scontig
  have da:=alloc.dataDescriptor.2.1
  have sa:=alloc.scaleDescriptor.2.1
  have rowsData : rows alloc.data.info=rows r.fpa.info:=by simp [rows,dshape,RowRequest.G,ComposedRequest.row]
  have colsData : cols alloc.data.info=cols r.fpa.info:=by simp [cols,dshape,RowRequest.K,ComposedRequest.row]
  have cuda:=r.contract.1.input.cuda
  have aTensor : TensorContract alloc.data.info [.other "int8"] 2 Bool.false := by
    constructor <;> simp [dshape,dtype,dlayout,ddevice,dneg,dcontig,bytes,RowRequest.G,RowRequest.K,ComposedRequest.row,cuda,Nat.mod_one]
  have ast : TensorContract alloc.scales.info [.other "float16",.floating .fp32] 1 Bool.false := by
    refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
    all_goals try simp [sshape,stype,slayout,sdevice,sneg,scontig,bytes,RowRequest.G,ComposedRequest.row,cuda]
    intro ne
    simpa [bytes,stype] using alloc.scaleAlignment ne
  have dataSeparate : ∀ t,t∈r.views → alloc.data.info.storage≠t.info.storage := by
    intro t ht eq;have ht:=inputs t ht;rw [←eq] at ht;exact alloc.dataFresh ht
  have scaleSeparate : ∀ t,t∈r.views → alloc.scales.info.storage≠t.info.storage := by
    intro t ht eq;have ht:=inputs t ht;rw [←eq] at ht;exact alloc.scaleFresh ht
  have db:=dataSeparate r.b (by simp [ComposedRequest.views])
  have ds:=dataSeparate r.bScale (by simp [ComposedRequest.views])
  have sb:=scaleSeparate r.b (by simp [ComposedRequest.views])
  have ss:=scaleSeparate r.bScale (by simp [ComposedRequest.views])
  have bm : r.b.info∈Direct.inputs r.fpa.info none r.b.info r.bScale.info Bool.true:=by simp [Direct.inputs]
  have bsm : r.bScale.info∈Direct.inputs r.fpa.info none r.b.info r.bScale.info Bool.true:=by simp [Direct.inputs]
  have mixed:=h.mixedAlias r.b.info bm r.bScale.info bsm
  have mixedBack : r.bScale.info.metadata.dtype=r.b.info.metadata.dtype ∨ r.bScale.info.storage≠r.b.info.storage:=
    mixed.elim (fun eq=>Or.inl eq.symm) (fun neq=>Or.inr neq.symm)
  refine ⟨aTensor,h.bTensor,h.bScaleTensor,?_,?_,?_,?_,?_,?_⟩
  · intro _;exact ⟨alloc.scales.info,rfl,ast,by simpa [sshape,RowRequest.G,ComposedRequest.row,rowsData]⟩
  · simpa only [colsData] using h.shape
  · intro t ht
    change t∈[alloc.data.info,alloc.scales.info,r.b.info,r.bScale.info] at ht
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with eq|eq|eq|eq <;> subst t
    · rfl
    · simp [ddevice,sdevice]
    · simpa only [ddevice,ComposedRequest.row] using h.device r.b.info bm
    · simpa only [ddevice,ComposedRequest.row] using h.device r.bScale.info bsm
  · intro u hu v hv
    change u∈[alloc.data.info,alloc.scales.info,r.b.info,r.bScale.info] at hu
    change v∈[alloc.data.info,alloc.scales.info,r.b.info,r.bScale.info] at hv
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hu hv
    rcases hu with eq|eq|eq|eq <;> subst u <;> rcases hv with eq|eq|eq|eq <;> subst v
    all_goals first
      | exact Or.inl rfl
      | exact Or.inr alloc.distinct
      | exact Or.inr alloc.distinct.symm
      | exact Or.inr db
      | exact Or.inr db.symm
      | exact Or.inr ds
      | exact Or.inr ds.symm
      | exact Or.inr sb
      | exact Or.inr sb.symm
      | exact Or.inr ss
      | exact Or.inr ss.symm
      | exact mixed
      | exact mixedBack
  · intro t ht
    have oh:=h.output t ht
    refine ⟨oh.tensor,?_,?_,?_⟩
    · simpa only [rowsData] using oh.shape
    · simpa only [ddevice,ComposedRequest.row] using oh.device
    · intro u hu
      change u∈[alloc.data.info,alloc.scales.info,r.b.info,r.bScale.info] at hu
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hu
      have existsOut : ∃ out,r.out=some out ∧ out.info=t := by
        cases eo : r.out with
        | none=>simp [eo] at ht
        | some out=>exact ⟨out,rfl,Option.some.inj (by simpa [eo] using ht)⟩
      obtain ⟨out,eq,id⟩:=existsOut
      have owned : out∈r.views:=by simp [ComposedRequest.views,eq]
      rcases hu with eq|eq|eq|eq <;> subst u
      · simpa only [id] using (dataSeparate out owned).symm
      · simpa only [id] using (scaleSeparate out owned).symm
      · exact oh.separated r.b.info bm
      · exact oh.separated r.bScale.info bsm
  · rcases h.launch with empty|empty|bounds
    · exact Or.inl (rowsData.trans empty)
    · exact Or.inr (Or.inl empty)
    · right;right;intro c hc
      have bound:=bounds c hc
      simpa [tileContract_iff,rowsData,colsData,da,sa,offset,gm,gn,fk,bRowStride,bColStride] using bound

def canonicalAllocated (t : TensorView) (dtype : VeriTile.Torch.DType) (dims strides : List Nat)
    (device : VeriTile.Torch.Device) : TensorView :=
  {t with info:={t.info with metadata:=⟨dtype,.strided,device,dims,Bool.true,Bool.false,t.info.metadata.dataPtr⟩,offset:=0},strides:=strides}
theorem canonicalAllocated_eq (t : TensorView) (dtype : VeriTile.Torch.DType) (dims strides : List Nat)
    (device : VeriTile.Torch.Device) (h : FreshDescriptor t dtype dims strides device) :
    canonicalAllocated t dtype dims strides device=t := by
  rcases t with ⟨⟨metadata,storage,offset,trans⟩,object,oldStrides,physical⟩
  simp only [canonicalAllocated,TensorView.mk.injEq,TensorInfo.mk.injEq,true_and,and_true]
  exact ⟨⟨h.1.symm,h.2.1.symm⟩,h.2.2.symm⟩
def composedA (r : ComposedRequest) (t : TensorView) :=
  canonicalAllocated t (.other "int8") [rows r.fpa.info,cols r.fpa.info] [max (cols r.fpa.info) 1,1] r.fpa.info.metadata.device
def composedScale (r : ComposedRequest) (t : TensorView) :=
  canonicalAllocated t (.other "float16") [rows r.fpa.info] [1] r.fpa.info.metadata.device

def producedRequest (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : RowAllocation r.row caller live) : GemmRequest where
  a:=composedA r alloc.data
  aScale:=composedScale r alloc.scales
  b:=r.b
  bScale:=r.bScale
  out:=r.out
  contract:=by
    have ad : composedA r alloc.data=alloc.data:=canonicalAllocated_eq _ _ _ _ _ alloc.dataDescriptor
    have sd : composedScale r alloc.scales=alloc.scales:=canonicalAllocated_eq _ _ _ _ _ alloc.scaleDescriptor
    simpa only [ad,sd] using produced_admission r caller live inputs alloc

theorem producedRequest_original (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : RowAllocation r.row caller live) :
    (producedRequest r caller live inputs alloc).a=alloc.data ∧
      (producedRequest r caller live inputs alloc).aScale=alloc.scales :=
  ⟨canonicalAllocated_eq _ _ _ _ _ alloc.dataDescriptor,canonicalAllocated_eq _ _ _ _ _ alloc.scaleDescriptor⟩

#axiomsClean produced_admission
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
