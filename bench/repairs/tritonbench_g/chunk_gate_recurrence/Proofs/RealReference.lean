import bench.repairs.tritonbench_g.chunk_gate_recurrence.IO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelRealReference

/-! Independent Real input-array formula and output packaging. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference
open VeriTile.Triton FP.Structural KernelProgram
open scoped BigOperators
noncomputable section
variable {α : Type} [Inhabited α]

def real (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    (IO.program KernelRealReference.algebra kind c s seed p).Inputs →
      (IO.program KernelRealReference.algebra kind c s seed p).Outputs :=
  match kind with
  | .forward => fun a o => Fin.cases
      (fun j => KernelRealReference.Formula.forward
        (KernelRealReference.inputs (forwardInputs a)) seed j.1.val j.2.1 j.2.2.1)
      (fun i => Fin.elim0 i) o
  | .backward => fun a o => Fin.cases
      (fun j => KernelRealReference.Formula.reverse
        (KernelRealReference.inputs (backwardInputs a)) c.N (j.1.val+1) j.2.1 j.2.2.1)
      (Fin.cases (fun j => (∑ k : Fin 1024,
        KernelRealReference.Formula.reverse (KernelRealReference.inputs (backwardInputs a))
          c.N (j.1.val+1) (KernelRealReference.index k).1 (KernelRealReference.index k).2.1 *
            (backwardInputs a).saved j.1.val (KernelRealReference.index k) : ℝ))
        (Fin.cases (fun j => KernelRealReference.Formula.reverse
          (KernelRealReference.inputs (backwardInputs a)) c.N 0 j.1 j.2.1)
          (fun i => Fin.elim0 i))) o

#specNonCircular real avoiding [FP.Structural.exec, FP.Structural.run, KernelReference.forward, KernelReference.reverse]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Reference
