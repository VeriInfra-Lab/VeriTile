import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.RealReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Correct
open VeriTile.Triton FP.Structural KernelProgram
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open scoped BigOperators
noncomputable section
set_option maxRecDepth 8192

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics

def forward (a : KernelRealReference.Formula.Inputs 64 16) (hasSeed : Bool) (c : Nat) (e : Fin 64) (j : Fin 16) : ℝ :=
  KernelRealReference.Formula.initial a hasSeed e j * (∏ t ∈ Finset.range c, a.gate t) +
    ∑ t ∈ Finset.range c, a.source t e j * ∏ u ∈ Finset.Ico (t+1) c, a.gate u

def reverse (a : KernelRealReference.Formula.Inputs 64 16) (N c : Nat) (e : Fin 64) (j : Fin 16) : ℝ :=
  ∑ u ∈ Finset.Ico c N, a.incoming u e j * ∏ w ∈ Finset.Ico c u, a.gate w

def result (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    LogicalTensors.Inputs kind c → LogicalTensors.Outputs kind c :=
  match kind with
  | .forward => fun a =>
      { state := fun time key value =>
          forward (KernelRealReference.inputs (LogicalTensors.forwardArrays c s seed p a)) seed time.val key value }
  | .backward => fun a =>
      { state := fun time key value => reverse
          (KernelRealReference.inputs (LogicalTensors.backwardArrays c s seed p a)) c.N (time.val+1) key value
        gate := fun time => ∑ key : Fin 64, ∑ value : Fin 16,
          reverse (KernelRealReference.inputs (LogicalTensors.backwardArrays c s seed p a))
            c.N (time.val+1) key value * a.saved time key value
        initial := fun key value => reverse
          (KernelRealReference.inputs (LogicalTensors.backwardArrays c s seed p a)) c.N 0 key value }
end Mathematics


/-- Complete FP32 source program, with explicit forward/backward channels,
optional seed strides and all time iterations. Real interprets its arithmetic
mathematically. The output footprint includes exactly the stores performed. -/
def kernelIO (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) := LogicalTensors.io kind c s seed p

/-- Independent exclusive recurrence, reverse recurrence and full tile DG sum.
The original per-program scope is retained; N=0 makes no host kernel launch. -/
specification chunk_gate_recurrence_kernel_real (kind : Kind) (c : Shape)
    (s : SeedStrides) (seed : Bool) (p : Program c) (nonempty : 0<c.N) :
    Spec.Real (kernelIO kind c s seed p ⊨ Mathematics.result kind c s seed p) :=
  by
    cases kind with
    | forward => exact LogicalTensors.correct .forward c s seed p nonempty
    | backward => exact LogicalTensors.backward_correct c s seed p nonempty

#axiomsClean chunk_gate_recurrence_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Correct
