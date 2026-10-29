import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTypedIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidateStructure

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

namespace Candidate

/-- This is an equality of the actual flattened real executions. It does
not infer erasure preservation from the separate typed FP equivalence. -/
theorem typed_flat_exec (A : FlatAlloc) (gradient saved destination : Typed.Storage)
    (v : Bool) (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) (s : BlockState) :
    exec (A.flattenKernel (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel) s =
    exec (A.flattenKernel (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel) s := by
  cases v
  · rfl
  · unfold exec FlatAlloc.flattenKernel
    conv_lhs => rw [typed_body gradient saved destination Bool.true]
    conv_rhs => rw [typed_body gradient saved destination Bool.false]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

theorem typed_erased_projects (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgorithm? =
    .ok (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, full_typed_projects]
  simp

end Candidate

/-- Original and candidate share the same independently strided IO contract;
the selected kernel is always the complete typed quotation after erasure. -/
def fullVariantIO (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) : Masked2DKernelIO₂ :=
  { fullTypedErasedIO gradient saved destination GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 with
    kernel := (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType
    projection := Candidate.typed_erased_projects gradient saved destination v
      GradIn GradOut Out bm bn gm gn om on K D log causal fp16 }

/-- Both complete source variants satisfy the independent backward formula
with successful flat execution and the full unwritten-memory frame. The
candidate proof transports an explicitly proved real flat-execution equality. -/
specification ksoftmax_backward_full_variant_real_correctness
    (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    fullVariantIO gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 ⊨
      fun _ row g o i => formula K row log causal g o i := by
  intro A hd hr hc m n hg ho hb hscratch g o s hm hn hu hgr hor
  obtain ⟨t, he, hv, hf⟩ := ksoftmax_backward_full_typed_real_correctness
    gradient saved destination GradIn GradOut Out bm bn gm gn om on K D log causal fp16
    A hd hr hc m n hg ho hb hscratch g o s hm hn hu hgr hor
  refine ⟨t, ?_, hv, hf⟩
  exact (Candidate.typed_flat_exec A gradient saved destination v GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16 (A.flattenState s)).trans he

#axiomsClean Candidate.typed_body
#axiomsClean Candidate.typed_flat_exec
#axiomsClean Candidate.typed_erased_projects
#axiomsClean ksoftmax_backward_full_variant_real_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
