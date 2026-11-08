import bench.tritonbench_g.int8_matmul_quantization.IO
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerSafetyProgram

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Logical bounds cover every masked iteration of both original loops and
its unmasked row-scale store; safety does not require a one-block K bound. -/
theorem source_safe (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (s : State α) (bounds : RegionName→Nat)
    (hx : ∀ c : Fin p.columns,r.val*p.columns+c.val<bounds "x")
    (hq : ∀ c : Fin p.columns,r.val*p.columns+c.val<bounds "q")
    (hs : r.val<bounds "scales") :
    TraceSafe (KernelProducer.backend M integer8) bounds
      (Source.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p)))
      (KernelProducerGrid.atRow s r) := by
  apply KernelProducer.SourceSafety.source_safe M integer8 _ _ p.positive bounds
  constructor
  · intro k i active
    have col : i.2.1.val+k*p.width<p.columns := by
      change i.2.1.val<p.columns-k*p.width at active
      omega
    have bounded:=hx ⟨i.2.1.val+k*p.width,col⟩
    simpa [KernelProducer.MaxPass.inputOffset,KernelProducerGrid.parameters,
      KernelProducerGrid.atRow,Nat.mod_eq_of_lt r.isLt,Nat.add_assoc] using bounded
  · intro k i active
    have col : i.2.1.val+k*p.width<p.columns := by
      change i.2.1.val<p.columns-k*p.width at active
      omega
    have bounded:=hq ⟨i.2.1.val+k*p.width,col⟩
    simpa [KernelProducer.QuantizePass.outputOffset,KernelProducerGrid.parameters,
      KernelProducerGrid.atRow,Nat.mod_eq_of_lt r.isLt,Nat.add_assoc] using bounded
  · simpa [KernelProducerGrid.parameters,KernelProducerGrid.atRow] using hs

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
