import bench.tritonbench_g.triton_softmax.Proofs.RealIO
import bench.tritonbench_g.triton_softmax.Proofs.Candidate
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Complete typed source projected to real arithmetic. Every active input
and output address is specified; the mathematical contract includes success
and the full frame outside active writes. -/
def typedIO (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) : MaskedKernelIO₁ :=
  { softmaxIO out inp si so n B with
    kernel := (typedKernel input output v out inp si so n B).eraseDType
    projection := typed_supported input output v out inp si so n B }

/-- Both complete source variants implement the independently defined masked
stable-softmax formula. Symbolic positive block size, both independent strides,
all partial rows and both input/output floating storage choices are retained.
This is real-erasure correctness; the FP theorem retains every precision. -/
specification softmax_typed_real_correct (input output : Storage) (v : Bool)
    (out inp : RegionName) (si so n B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedIO input output v out inp si so n B ⊨
      fun xs i => softmaxSpec n B xs i) := by
  intro A hd hr hc pid hin hout hscratch xs s hpid hu hx
  obtain ⟨t, he, hv, hf⟩ := softmax_kernel_real_io out inp si so n B hB
    A hd hr hc pid hin hout hscratch xs s hpid hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A input output v out inp si so n B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax
