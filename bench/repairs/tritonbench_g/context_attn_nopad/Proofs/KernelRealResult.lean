import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Kernels
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.KernelRealResult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- Full shared source grid on raw shape/strides, integer input arrays, finite
allocation windows and initial memory. Every self-inclusive causal output,
input cell and exterior cell is covered; no actual Tensor occurs. -/
theorem correct (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (memory : Host.Memory c r A s) (disjoint : A.Disjoint)
    (covered : ∀ region,region∉A.regions → A.extent region=0) (complete : c.completeGrid) :
    ∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (kernel c r).eraseDType.toAlgKernel) c.grid (A.flattenState s) final) ∧
      (∀ b head t d,b<c.batches → head<c.q.heads → t<c.length b → d<c.q.dims →
        final.readMem A.flat (A.addr r.out (c.output b head t d))=
          c.expected (inputs s r) b head t d) ∧
      BlockState.WriteWithin (c.writes r A s) (A.flattenState s) final ∧
      (∀ region∈c.inputRegions r,∀ offset,offset<A.extent region →
        final.mem A.flat (A.addr region offset)=(A.flattenState s).mem A.flat (A.addr region offset)) := by
  obtain ⟨result,run,values,frame,stable⟩:=Host.host_complete memory disjoint covered complete Bool.false
  exact ⟨result.1,run.1,values,frame,stable⟩

#axiomsClean correct
#specNonCircular Call.expected avoiding [kernel,VeriTile.Triton.exec]
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.KernelRealResult
