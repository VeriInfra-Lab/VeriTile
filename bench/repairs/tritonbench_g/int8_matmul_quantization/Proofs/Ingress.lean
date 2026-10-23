import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Domain
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.QuantReference

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
variable {s : Shape}

theorem produced_A (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate) (b : Buffers)
    (link : CompositionLink q b) (calls : List (QuantInvocation s.M s.K))
    (chosen : QuantInvocation s.M s.K) (m : Memory F H) (r : Fin s.M) (k : Fin s.K) :
    read8 (quantWrapper a q calls chosen m) (b.a,b.aOffset+r.val*s.K+k.val)=quantValue a q chosen.plan m (r,k) := by
  have addr : (b.a,b.aOffset+r.val*s.K+k.val)=q.dataAddress (r,k) := by
    simp [QuantBuffers.dataAddress,link.aStorage,link.aOffset,link.rowLayout,Nat.add_assoc]
  rw [addr]
  have h := quantWrapper_at a q sep calls chosen m (.inl (r,k))
  change quantWrapper a q calls chosen m (q.dataAddress (r,k))=.inl _ at h
  simp only [read8,Int8DequantMatmul.read8,h]
theorem produced_scale (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate) (b : Buffers)
    (link : CompositionLink q b) (calls : List (QuantInvocation s.M s.K))
    (chosen : QuantInvocation s.M s.K) (m : Memory F H) (r : Fin s.M) :
    loadScale a b.aScalePrecision (quantWrapper a q calls chosen m) (b.aScale,b.aScaleOffset+r.val)=
      a.widen (a.narrow (quantScale a q chosen.plan m r)) := by
  have addr : (b.aScale,b.aScaleOffset+r.val)=q.scaleAddress r := by
    simp [QuantBuffers.scaleAddress,link.scaleStorage,link.scaleOffset]
  rw [link.scaleType,addr]
  have h := quantWrapper_at a q sep calls chosen m (.inr r)
  simp only [QuantBuffers.writeAddress,quantExpected,link.rowScaleType] at h
  simp only [loadScale,read16,Int8DequantMatmul.read16,h]
theorem compiled_produced_A (abi : LocalEventABI I F H σ a) (q : QuantBuffers) (sep : q.Separate)
    (b : Buffers) (link : CompositionLink q b)
    (calls : List (BoundQuantCall I F H s.M s.K q)) (chosen : BoundQuantCall I F H s.M s.K q)
    (state : σ) (r : Fin s.M) (k : Fin s.K) :
    read8 (abi.observe (compiledQuantWrapper abi calls chosen state)) (b.a,b.aOffset+r.val*s.K+k.val)=
      quantValue a q chosen.invocation.plan (abi.observe state) (r,k) := by
  rw [compiledQuantWrapper_refines]
  exact produced_A a q sep b link _ _ _ r k
#axiomsClean produced_A
#axiomsClean produced_scale
#axiomsClean compiled_produced_A
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
