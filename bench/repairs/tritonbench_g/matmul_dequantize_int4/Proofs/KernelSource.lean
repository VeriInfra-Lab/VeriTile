import bench.repairs.tritonbench_g.matmul_dequantize_int4.Kernels
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GridExecution

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution
noncomputable section

/-- The readable shared `triton` quotation retains the selected source's
signed packed loads, both group branches, loop, source Dot+Add and all stores.
Implicit half promotions and the destination conversion are materialized. -/
abbrev kernel (c : Call) (i : Fin 17) : ComputeKernel := source (params c i)

def count (c : Call) (i : Fin 17) : Nat :=
  blocksM (params c i)*blocksN (params c i)

def one {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Call) (i : Fin 17) (pid : Nat) (s : State α) : Option (State α) :=
  FP.Structural.exec (backend M D) (kernel c i) (atProgram s pid)

def run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Call) (i : Fin 17) (s : State α) : Option (State α) :=
  sequence (one M D c i) (List.range (count c i)) s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelSource
