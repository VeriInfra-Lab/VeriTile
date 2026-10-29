import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantContract
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.TensorBinding
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantTensor
open VeriTile.Triton Checked
noncomputable section
abbrev Heap := TensorBinding.Heap

/-- Signed int32 Tensor object, with ordinary allocation and coordinate laws. -/
structure PackedTensor where
  observed : Observation
  view : View
  tensor : TensorBinding.Tensor observed view
  dtype : observed.metadata.dtype=.other "int32"

/-- Ordinary public allocator/transpose-contiguous API result. Input is the
actual shared FP32 Tensor; its stride coordinates are observed, not copied. -/
structure Tensors (p : QuantMetadata) where
  weight : VeriTile.Torch.FP32.Tensor
  weightLaws : VeriTile.Torch.FP32.TensorLaws weight
  weightMetadata : weight.metadata=p.weight.metadata
  weightObject : weight.object=p.weight.object
  weightStorage : weight.storage.object=p.weight.storageObject
  weightBase : weight.storage.base=p.weight.storageBase
  weightBytes : 4*weight.storage.elements=p.weight.storageBytes
  weightOffset : weight.storageOffset=p.weight.storageOffset
  weightCoordinates : ∀ k n,k<p.call.shape.K → n<p.call.shape.N →
    weight.physicalIndex (k*p.call.shape.N+n)=weight.storageOffset+p.call.W.addr k n
  (q z : PackedTensor)
  s : VeriTile.Torch.FP32.Tensor
  sLaws : VeriTile.Torch.FP32.TensorLaws s
  qShape : q.observed.metadata.shape=[p.call.shape.K/8,p.call.shape.N]
  zShape : z.observed.metadata.shape=[p.call.shape.groups,p.call.shape.N/8]
  sShape : s.metadata.shape=[p.call.shape.groups,p.call.shape.N]
  qContiguous : q.observed.metadata.isContiguous=Bool.true
  zContiguous : z.observed.metadata.isContiguous=Bool.true
  sContiguous : s.metadata.isContiguous=Bool.true
  qView : q.view.rows=p.call.shape.K/8 ∧ q.view.cols=p.call.shape.N ∧ q.view.sr=p.call.shape.N ∧ q.view.sc=1
  zView : z.view.rows=p.call.shape.groups ∧ z.view.cols=p.call.shape.N/8 ∧ z.view.sr=p.call.shape.N/8 ∧ z.view.sc=1
  qOffset : q.observed.storageOffset=0
  zOffset : z.observed.storageOffset=0
  sOffset : s.storageOffset=0
  qDevice : q.observed.metadata.device=⟨"cuda",p.call.tpRank⟩
  zDevice : z.observed.metadata.device=⟨"cuda",p.call.tpRank⟩
  sDevice : s.metadata.device=⟨"cuda",p.call.tpRank⟩
  distinctQS : q.observed.storageObject≠s.storage.object
  distinctQZ : q.observed.storageObject≠z.observed.storageObject
  distinctSZ : s.storage.object≠z.observed.storageObject
  freshQ : q.observed.storageObject≠weight.storage.object
  freshS : s.storage.object≠weight.storage.object
  freshZ : z.observed.storageObject≠weight.storage.object

def memory (p : QuantMetadata) (t : Tensors p) (heap : Heap) : Physical.Bytes := fun r b =>
  if r="W" then heap t.weight.storage.object (4*t.weight.storageOffset+b)
  else if r="Qout" then heap t.q.observed.storageObject b
  else if r="Sout" then heap t.s.storage.object b
  else if r="Zout" then heap t.z.observed.storageObject b else 0

theorem raw_input (p : QuantMetadata) (t : Tensors p) (heap : Heap) (k n : Nat)
    (hk : k<p.call.shape.K) (hn : n<p.call.shape.N) :
    QuantHost.input p.call (memory p t heap) k n=
      ⟨Physical.decode32 (fun byte => heap t.weight.storage.object
        (4*t.weight.physicalIndex (k*p.call.shape.N+n)+byte.val))⟩ := by
  simp only [QuantHost.input,memory,ite_true,t.weightCoordinates k n hk hn]
  congr 2
  funext byte
  congr 1
  omega

def physical (p : QuantMetadata) (t : Tensors p) (heap : Heap) (m : Physical.Bytes) : Heap := fun r b =>
  if r=t.q.observed.storageObject then m "Qout" b
  else if r=t.s.storage.object then m "Sout" b
  else if r=t.z.observed.storageObject then m "Zout" b else heap r b

structure Result where
  memory : Heap
  returned : PackedTensor × VeriTile.Torch.FP32.Tensor × PackedTensor × Nat

def invoke (p : QuantMetadata) (t : Tensors p) (actual : QuantFP.Instructions) (heap : Heap) : Except Error Result := do
  checkQuant p (memory p t heap)
  let result := QuantHost.call p.call actual (memory p t heap)
  pure ⟨physical p t heap result.memory,(t.q,t.s,t.z,p.call.shape.group)⟩

def Writes (p : QuantMetadata) (t : Tensors p) (r b : Nat) : Prop :=
  (r=t.q.observed.storageObject ∧ b/4<(p.call.shape.K/8)*p.call.shape.N) ∨
  (r=t.s.storage.object ∧ b/4<p.call.shape.groups*p.call.shape.N) ∨
  (r=t.z.observed.storageObject ∧ b/4<p.call.shape.groups*(p.call.shape.N/8))

def Guarantee (p : QuantMetadata) (t : Tensors p) (heap : Heap) (result : Result) : Prop :=
  result.returned=(t.q,t.s,t.z,p.call.shape.group) ∧
  (∀ k n,k<p.call.shape.K/8 → n<p.call.shape.N → ∀ byte : Fin 4,
    result.memory t.q.observed.storageObject (4*(k*p.call.shape.N+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetPackedQ (QuantFP.ops QuantFP.software p.call.tree) p.call.shape
        (QuantHost.input p.call (memory p t heap)) k n) byte) ∧
  (∀ g n,g<p.call.shape.groups → n<p.call.shape.N → ∀ byte : Fin 4,
    result.memory t.s.storage.object (4*(g*p.call.shape.N+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetScale (QuantFP.ops QuantFP.software p.call.tree) p.call.shape
        (QuantHost.input p.call (memory p t heap)) g n).bits byte) ∧
  (∀ g n,g<p.call.shape.groups → n<p.call.shape.N/8 → ∀ byte : Fin 4,
    result.memory t.z.observed.storageObject (4*(g*(p.call.shape.N/8)+n)+byte.val)=
      QuantHost.encode32 (Quantizer.targetPackedZ (QuantFP.ops QuantFP.software p.call.tree) p.call.shape
        (QuantHost.input p.call (memory p t heap)) g n) byte) ∧
  (∀ r b,¬Writes p t r b → result.memory r b=heap r b)

theorem whole_tensor (p : QuantMetadata) (t : Tensors p) (heap : Heap)
    (h : QuantInputContract p (memory p t heap)) (tree : ReductionBinding p.call)
    (actual : QuantFP.Instructions) (ports : QuantFP.Obligations actual) :
    ∃ result,invoke p t actual heap=.ok result ∧ Guarantee p t heap result := by
  have admitted := quant_admitted p (memory p t heap) h tree
  have result := QuantHost.whole_call p.call admitted.valid actual ports (memory p t heap)
  let raw := QuantHost.call p.call actual (memory p t heap)
  refine ⟨⟨physical p t heap raw.memory,(t.q,t.s,t.z,p.call.shape.group)⟩,?_,rfl,?_,?_,?_,?_⟩
  · simp [invoke,(checkQuant_iff _ _).mpr h,raw]
  · intro k n hk hn byte
    simpa only [physical,ite_true] using result.1 k n hk hn byte
  · intro g n hg hn byte
    simpa only [physical,Ne.symm t.distinctQS,ite_false,ite_true] using result.2.1 g n hg hn byte
  · intro g n hg hn byte
    simpa only [physical,Ne.symm t.distinctQZ,Ne.symm t.distinctSZ,ite_false,ite_true] using result.2.2.1 g n hg hn byte
  · intro r b outside
    by_cases hq : r=t.q.observed.storageObject
    · have no : ¬QuantHost.Writes p.call "Qout" b := by
        simp only [QuantHost.Writes]
        simp only [Writes,hq,true_and] at outside
        simp only [true_and,show ("Qout":RegionName)≠"Sout" by decide,
          show ("Qout":RegionName)≠"Zout" by decide,false_and,or_false]
        exact fun h => outside (Or.inl h)
      simp only [physical,hq,ite_true]
      exact (result.2.2.2.1 "Qout" b no).trans (by simpa only [hq] using (show memory p t heap "Qout" b=heap t.q.observed.storageObject b by simp [memory]))
    · by_cases hs : r=t.s.storage.object
      · have no : ¬QuantHost.Writes p.call "Sout" b := by
          simp only [QuantHost.Writes]
          simp only [Writes,hs,true_and] at outside
          simp only [true_and,show ("Sout":RegionName)≠"Qout" by decide,
            show ("Sout":RegionName)≠"Zout" by decide,false_and,false_or,or_false]
          exact fun h => outside (Or.inr (Or.inl h))
        simp only [physical,hq,hs,Ne.symm t.distinctQS,ite_true,ite_false]
        exact (result.2.2.2.1 "Sout" b no).trans (by simpa only [hs] using (show memory p t heap "Sout" b=heap t.s.storage.object b by simp [memory]))
      · by_cases hz : r=t.z.observed.storageObject
        · have no : ¬QuantHost.Writes p.call "Zout" b := by
            simp only [QuantHost.Writes]
            simp only [Writes,hz,true_and] at outside
            simp only [true_and,show ("Zout":RegionName)≠"Qout" by decide,
              show ("Zout":RegionName)≠"Sout" by decide,false_and,false_or]
            exact fun h => outside (Or.inr (Or.inr h))
          simp only [physical,hq,hs,hz,Ne.symm t.distinctQZ,Ne.symm t.distinctSZ,ite_true,ite_false]
          exact (result.2.2.2.1 "Zout" b no).trans (by simpa only [hz] using (show memory p t heap "Zout" b=heap t.z.observed.storageObject b by simp [memory]))
        · simp [physical,hq,hs,hz]

#specNonCircular Guarantee avoiding [invoke,QuantHost.call,Quantizer.source,QuantHost.publish]
#axiomsClean raw_input
#axiomsClean whole_tensor
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantTensor
