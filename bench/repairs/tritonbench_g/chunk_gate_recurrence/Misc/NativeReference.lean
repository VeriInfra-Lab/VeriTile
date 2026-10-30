import bench.repairs.tritonbench_g.chunk_gate_recurrence.IO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelWords

/-! Optional native observation target, retaining the selected FMA order and
complete 1024-product reduction plan, including its fused leaves. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
open scoped BigOperators
noncomputable section
variable {α : Type} [Inhabited α]

def native (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    (IO.program (model isa plan kind c) kind c s seed p).Inputs →
      (IO.program (model isa plan kind c) kind c s seed p).ObservedOutputs Word :=
  match kind with
  | .forward => fun a o => Fin.cases
      (fun j => KernelWords.forward isa (swapped .forward c)
        (KernelWords.inputs (forwardInputs a)) seed j.1.val j.2)
      (fun i => Fin.elim0 i) o
  | .backward => fun a o => Fin.cases
      (fun j => KernelWords.reverse isa (swapped .backward c)
        (KernelWords.inputs (backwardInputs a)) c.N (c.N-1-j.1.val) j.2)
      (Fin.cases (fun j => KernelWords.contraction isa plan
        (KernelWords.reverse isa (swapped .backward c)
          (KernelWords.inputs (backwardInputs a)) c.N (c.N-1-j.1.val))
        ((KernelWords.inputs (backwardInputs a)).saved j.1.val))
        (Fin.cases (fun j => KernelWords.reverse isa (swapped .backward c)
          (KernelWords.inputs (backwardInputs a)) c.N c.N j)
          (fun i => Fin.elim0 i))) o

#specNonCircular native avoiding [FP.Structural.exec, FP.Structural.run, KernelReference.forward, KernelReference.reverse]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference
