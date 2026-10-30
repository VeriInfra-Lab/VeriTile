import bench.repairs.tritonbench_g.matmul_leakyrelu.Contract

/- The source launcher selects a.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole

theorem indexChecks_iff (p : Metadata) (activation : String) :
    indexChecks p = Bool.true ↔ IndexDomain (request p activation) := by
  simp only [indexChecks, decide_eq_true_eq]
  constructor
  · rintro ⟨hm,hn,hk,ham,hak,hbk,hbn,hg,hw,ha,hb,hc⟩
    exact ⟨hm,hn,hk,ham,hak,hbk,hbn,hg,hw,ha,hb,hc⟩
  · intro h
    exact ⟨h.rows,h.columns,h.inner,h.strideAM,h.strideAK,h.strideBK,h.strideBN,
      h.grid,h.group,h.paddedA,h.paddedB,h.paddedC⟩

theorem checkInputs_iff (p : Metadata) : checkInputs p = .ok () ↔ InputContract p := by
  have hc : InputContract p ↔ metadataChecks p = Bool.true ∧ IndexDomain (request p "") := by
    constructor
    · intro h; exact ⟨h.metadata,h.indices⟩
    · rintro ⟨hm,hi⟩; exact ⟨hm,hi⟩
  rw [hc, ← indexChecks_iff]
  simp [checkInputs, ite_eq_iff]

theorem checkArguments_iff (args : Arguments) : checkArguments args = .ok () ↔ Admitted args := by
  cases args <;> simp [checkArguments, Admitted, checkInputs_iff]

theorem admitted_indices (p : Metadata) (h : InputContract p) (activation : String) :
    IndexDomain (request p activation) :=
  (indexChecks_iff p activation).1 ((indexChecks_iff p "").2 h.indices)

/-- Ordinary metadata, storage, observation and fresh-allocation facts. Input
alignment is derived from the executed guard; empty Tensor.data_ptr() is zero. -/
structure OrdinaryTensorContract {α : Type} (p : Metadata) (activation : String)
    (s : State α) where
  (extentA extentB extentC : Nat)
  (baseA baseB baseC : Nat)
  (metadataA metadataB metadataC : TensorMetadata)
  shapeA : metadataA.shape=[(request p activation).rows,(request p activation).inner]
  shapeB : metadataB.shape=[(request p activation).innerB,(request p activation).columns]
  shapeC : metadataC.shape=[(request p activation).rows,(request p activation).columns]
  stridesA : metadataA.strides=[(request p activation).strideAM,(request p activation).strideAK]
  stridesB : metadataB.strides=[(request p activation).strideBK,(request p activation).strideBN]
  stridesC : metadataC.strides=[max (request p activation).columns 1,1]
  dtypeA : metadataA.dtype=.fp32
  dtypeB : metadataB.dtype=.fp32
  dtypeC : metadataC.dtype=.fp16
  deviceB : metadataB.device=metadataA.device
  deviceC : metadataC.device=metadataA.device
  dimensions : (request p activation).inner=(request p activation).innerB
  contiguousA : (request p activation).aContiguous=Bool.true
  freshA : (request p activation).C≠(request p activation).A
  freshB : (request p activation).C≠(request p activation).B
  physicalFreshA : byteSpansDisjoint baseC (2*extentC) baseA (4*extentA)
  physicalFreshB : byteSpansDisjoint baseC (2*extentC) baseB (4*extentB)
  alignmentC : baseC%2=0
  spanA : baseA+4*extentA≤2^64
  spanB : baseB+4*extentB≤2^64
  spanC : baseC+2*extentC≤2^64
  sameViewOrigin : (request p activation).A=(request p activation).B → baseA=baseB
  aliasConsistency : ∀ row<(request p activation).rows, ∀ k<(request p activation).inner, ∀ k'<(request p activation).inner, ∀ col<(request p activation).columns,
    baseA+4*(row*(request p activation).strideAM+k*(request p activation).strideAK)=baseB+4*(k'*(request p activation).strideBK+col*(request p activation).strideBN) →
    s.mem (request p activation).A (row*(request p activation).strideAM+k*(request p activation).strideAK)=s.mem (request p activation).B (k'*(request p activation).strideBK+col*(request p activation).strideBN)
  inputA : ∀ row<(request p activation).rows, ∀ k<(request p activation).inner,
    row*(request p activation).strideAM+k*(request p activation).strideAK<extentA
  inputB : ∀ k<(request p activation).inner, ∀ col<(request p activation).columns,
    k*(request p activation).strideBK+col*(request p activation).strideBN<extentB
  output : ∀ row<(request p activation).rows, ∀ col<(request p activation).columns, outputOffset (request p activation) row col<extentC
  tagA : ∀ row<(request p activation).rows, ∀ k<(request p activation).inner,
    ∃ x, s.mem (request p activation).A (row*(request p activation).strideAM+k*(request p activation).strideAK)=Cell.mk .fp32 x
  tagB : ∀ k<(request p activation).inner, ∀ col<(request p activation).columns,
    ∃ x, s.mem (request p activation).B (k*(request p activation).strideBK+col*(request p activation).strideBN)=Cell.mk .fp32 x
  pointerA : baseA=p.a.dataPtr.toNat
  pointerB : baseB=p.b.dataPtr.toNat
  emptyA : p.a.numel=0 → p.a.dataPtr.toNat=0
  emptyB : p.b.numel=0 → p.b.dataPtr.toNat=0

def OrdinaryTensorContract.admitted {α : Type} {p : Metadata} {activation : String}
    {s : State α} (ordinary : OrdinaryTensorContract p activation s)
    (admitted : InputContract p) (output : outputCheck p ordinary.baseC=Bool.true) : TensorContract (request p activation) s := by
  have hm := admitted.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  obtain ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,alignA,alignB,byteA,byteB⟩ := hm
  have ha : p.a.dataPtr.toNat%4=0 := by
    rcases alignA with empty | aligned
    · simp [ordinary.emptyA empty]
    · exact aligned
  have hb : p.b.dataPtr.toNat%4=0 := by
    rcases alignB with empty | aligned
    · simp [ordinary.emptyB empty]
    · exact aligned
  exact {
    extentA := ordinary.extentA
    extentB := ordinary.extentB
    extentC := ordinary.extentC
    baseA := ordinary.baseA
    baseB := ordinary.baseB
    baseC := ordinary.baseC
    metadataA := ordinary.metadataA
    metadataB := ordinary.metadataB
    metadataC := ordinary.metadataC
    shapeA := ordinary.shapeA
    shapeB := ordinary.shapeB
    shapeC := ordinary.shapeC
    stridesA := ordinary.stridesA
    stridesB := ordinary.stridesB
    stridesC := ordinary.stridesC
    dtypeA := ordinary.dtypeA
    dtypeB := ordinary.dtypeB
    dtypeC := ordinary.dtypeC
    deviceB := ordinary.deviceB
    deviceC := ordinary.deviceC
    dimensions := ordinary.dimensions
    contiguousA := ordinary.contiguousA
    freshA := ordinary.freshA
    freshB := ordinary.freshB
    physicalFreshA := ordinary.physicalFreshA
    physicalFreshB := ordinary.physicalFreshB
    alignmentC := ordinary.alignmentC
    spanA := ordinary.spanA
    spanB := ordinary.spanB
    spanC := ordinary.spanC
    sameViewOrigin := ordinary.sameViewOrigin
    aliasConsistency := ordinary.aliasConsistency
    inputA := ordinary.inputA
    inputB := ordinary.inputB
    output := ordinary.output
    tagA := ordinary.tagA
    tagB := ordinary.tagB
    byteA := by simpa only [ordinary.pointerA,request] using byteA
    byteB := by simpa only [ordinary.pointerB,request] using byteB
    byteC := by simpa only [outputCheck,decide_eq_true_eq,request] using output
    alignmentA := by rw [ordinary.pointerA]; exact ha
    alignmentB := by rw [ordinary.pointerB]; exact hb }

#axiomsClean checkArguments_iff
#axiomsClean admitted_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulLeakyrelu
