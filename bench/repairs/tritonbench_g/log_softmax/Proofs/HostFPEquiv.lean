import bench.repairs.tritonbench_g.log_softmax.Proofs.Host
import bench.repairs.tritonbench_g.log_softmax.Proofs.Machine
import bench.repairs.tritonbench_g.log_softmax.Proofs.FP

namespace VeriTile.Bench.Repairs.LogSoftmax.HostFPEquiv
open VeriTile.Torch Address Execution Host Native
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open Classical

/-- Whole native forward call for all FP32 words. Reference rows expand the
selected padded max/sum trees, and local scalar ports pin rounded operations.
Shape/dtype/device, saved-output identity, empty branches and byte frames are
unconditional structural results under the ordinary memory/library contracts. -/
theorem log_softmax_fp32 (api : API Word Store Byte) (actual : Scalars) (target : Target)
    (ports : LocalObligations actual target) (a : Arguments) (h : InputContract a)
    (b : Buffers a.M a.N a.K) (probes : List (Grid.Selected a.M a.N a.K))
    (selected : Grid.Selected a.M a.N a.K) (s : Store) :
    ∃ result,forwardCall api (ops actual) a b probes selected s=.ok result ∧
      result.shape=a.input.shape ∧ result.dtype=.floating .fp32 ∧ result.device=a.input.device ∧
      result.output=b.output ∧ result.context.saved=result.output ∧ result.context.dim=a.axis ∧
      (∀ q : Coord a.M a.N a.K,api.read result.memory (result.output.address (linear q))=
        forward (ops (reference target)) selected.plan (rowRead api b.first s q) q.2.1.val) ∧
      (∀ i,¬b.output.contains i → api.byte result.memory i=api.byte s i) := by
  refine ⟨⟨dispatch api (ops actual) .forward b probes selected s,a.input.shape,.floating .fp32,
    a.input.device,b.output,⟨b.output,a.axis⟩⟩,forward_success api (ops actual) a h b probes selected s,
    rfl,rfl,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro q
    rw [dispatch_values]
    exact forward_equiv actual target ports selected.plan _ _
  · exact dispatch_frame api (ops actual) .forward b probes selected s

/-- Backward uses the actual saved output words. Its schedule retains either
separate multiply/subtract or either contracted FMA operand order; no equality
between those schedules is presumed. The API return tuple and full frame are
proved together with all gradient cells. -/
theorem log_softmax_backward_fp32 (api : API Word Store Byte) (actual : Scalars) (target : Target)
    (ports : LocalObligations actual target) (a : BackwardArguments) (h : BackwardContract a)
    (ctx : Context a.saved) (b : Buffers a.saved.M a.saved.N a.saved.K)
    (saved : ctx.saved=b.first) (dimension : ctx.dim=a.saved.axis)
    (probes : List (Grid.Selected a.saved.M a.saved.N a.saved.K))
    (selected : Grid.Selected a.saved.M a.saved.N a.saved.K) (s : Store) :
    ∃ result,backwardCall api (ops actual) a ctx b saved dimension probes selected s=.ok result ∧
      result.shape=a.saved.input.shape ∧ result.dtype=.floating .fp32 ∧ result.device=a.saved.input.device ∧
      result.output=b.output ∧ result.dimGradient=none ∧ result.dtypeGradient=none ∧
      (∀ q : Coord a.saved.M a.saved.N a.saved.K,api.read result.memory (result.output.address (linear q))=
        backward (ops (reference target)) selected.plan (rowRead api ctx.saved s q)
          (rowRead api b.second s q) q.2.1.val) ∧
      (∀ i,¬b.output.contains i → api.byte result.memory i=api.byte s i) := by
  refine ⟨⟨dispatch api (ops actual) .backward b probes selected s,a.saved.input.shape,.floating .fp32,
    a.saved.input.device,b.output,none,none⟩,
    backward_success api (ops actual) a h ctx b saved dimension probes selected s,
    rfl,rfl,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro q
    rw [dispatch_values,saved]
    exact backward_equiv actual target ports selected.plan _ _ _
  · exact dispatch_frame api (ops actual) .backward b probes selected s

#axiomsClean log_softmax_fp32
#axiomsClean log_softmax_backward_fp32
end
end VeriTile.Bench.Repairs.LogSoftmax.HostFPEquiv
