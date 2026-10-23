import bench.repairs.tritonbench_g.matmul_dequantize_int4.Contract
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Checked
open VeriTile.Triton
noncomputable section

def quantConfigurationChecks (c : QuantHost.Call) : Bool := decide (
  0<c.shape.K ∧ 0<c.shape.N ∧ 0<c.shape.group ∧ 8∣c.shape.K ∧ 8∣c.shape.N ∧ c.shape.group∣c.shape.K ∧
  c.W.rows=c.shape.K ∧ c.W.cols=c.shape.N ∧ viewChecks c.W ∧
  4*c.W.capacity<2^63 ∧ 4*(c.shape.K*c.shape.N)<2^63)

/-- Library reduction routing is structural execution data. This is separate
from admission: Python does not inspect the CUDA library's reduction tree. -/
structure ReductionBinding (c : QuantHost.Call) : Prop where
  legal : c.tree.Valid c.shape.group

def positive (w : QuantFP.Word) : Bool := match w.toRat? with
  | none => Bool.false | some q => decide (0<q)
theorem positive_iff (w : QuantFP.Word) : positive w=Bool.true ↔ ∃ q,0<q ∧ w.toRat?=some q := by
  unfold positive
  cases he : w.toRat? <;> simp [he]

def quantFiniteChecks (c : QuantHost.Call) (m : Physical.Bytes) : Bool :=
  let x := QuantHost.input c m
  let ops := QuantFP.ops QuantFP.software c.tree
  let scale := Quantizer.targetScale ops c.shape x
  let zero := Quantizer.targetZero ops c.shape x
  ((List.range c.shape.K).all fun k => (List.range c.shape.N).all fun n => (x k n).toRat?.isSome) &&
  ((List.range c.shape.groups).all fun g => (List.range c.shape.N).all fun n => positive (scale g n)) &&
  ((List.range c.shape.groups).all fun g => (List.range c.shape.N).all fun n =>
    (FP.div .ieeeValue (FP.Value.neg (Quantizer.low ops c.shape.group (Quantizer.groupValues c.shape x g n)))
      (scale g n)).toRat?.isSome) &&
  ((List.range c.shape.K).all fun k => (List.range c.shape.N).all fun n =>
    (FP.div .ieeeValue (x k n) (scale (k/c.shape.group) n)).toRat?.isSome &&
    (FP.add .ieeeValue (FP.div .ieeeValue (x k n) (scale (k/c.shape.group) n))
      (ops.fromNibble (zero (k/c.shape.group) n))).toRat?.isSome)

theorem quantFiniteChecks_iff (c : QuantHost.Call) (m : Physical.Bytes) :
    quantFiniteChecks c m=Bool.true ↔ QuantFP.FiniteDomain c.shape (QuantHost.input c m) c.tree := by
  simp only [quantFiniteChecks,Bool.and_eq_true,List.all_eq_true,List.mem_range,
    Option.isSome_iff_exists,positive_iff,QuantFP.FiniteDomain]
  aesop

structure QuantMetadata where
  call : QuantHost.Call
  weight : Observation

def quantMetadataChecks (p : QuantMetadata) : Bool := decide (
  p.weight.metadata.dtype=.floating .fp32 ∧ p.weight.metadata.layout=.strided ∧
  p.weight.metadata.shape=[p.call.W.rows,p.call.W.cols] ∧
  p.weight.metadata.isNeg=Bool.false ∧ p.weight.isConj=Bool.false ∧
  p.weight.metadata.dataPtr.toNat=p.weight.storageBase+4*p.weight.storageOffset ∧
  p.weight.metadata.dataPtr.toNat%4=0 ∧ p.weight.storageBase+p.weight.storageBytes<2^64 ∧
  p.weight.storageBytes%4=0 ∧ 4*(p.weight.storageOffset+p.call.W.capacity)≤p.weight.storageBytes ∧
  p.weight.storageBytes<2^63)

def checkQuant (p : QuantMetadata) (m : Physical.Bytes) : Except Error Unit :=
  if !quantMetadataChecks p then .error .metadata
  else if !quantConfigurationChecks p.call then .error .shape
  else if !quantFiniteChecks p.call m then .error .nonfinite else .ok ()

def QuantInputContract (p : QuantMetadata) (m : Physical.Bytes) : Prop :=
  quantMetadataChecks p=Bool.true ∧ quantConfigurationChecks p.call=Bool.true ∧ quantFiniteChecks p.call m=Bool.true

theorem checkQuant_iff (p : QuantMetadata) (m : Physical.Bytes) :
    checkQuant p m=.ok () ↔ QuantInputContract p m := by
  simp [checkQuant,QuantInputContract,ite_eq_iff]

theorem quant_admitted (p : QuantMetadata) (m : Physical.Bytes) (h : QuantInputContract p m)
    (tree : ReductionBinding p.call) : Public.QuantAdmitted p.call m := by
  have hc := h.2.1
  simp only [quantConfigurationChecks,decide_eq_true_eq] at hc
  obtain ⟨k,n,g,k8,n8,groups,wk,wn,v,wi,copy⟩ := hc
  exact ⟨⟨⟨k,n,g,k8,n8,groups⟩,⟨wk,wn⟩,view_valid _ v,tree.legal⟩,
    ⟨wi,copy⟩,(quantFiniteChecks_iff _ _).mp h.2.2⟩

#specNonCircular QuantInputContract avoiding [QuantHost.call,Quantizer.source,Quantizer.sourceQ,Quantizer.sourceScale]
#axiomsClean quantFiniteChecks_iff
#axiomsClean checkQuant_iff
#axiomsClean quant_admitted
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Checked
