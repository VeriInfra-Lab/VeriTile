import bench.tritonbench_g.matmul_triton_autotune.Kernels
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FP

/-! Complete selected FP32 source, actual CUDA/HIP configs, ceil-K, arbitrary
activation string, tensor strides, fresh FP16 output, grouped launch and return.
Only primitive numerical conformance is deferred: live-carry dotAcc (including
TF32/default dot precision), ordered activation and output conversions/specials.
The theorem does not assert equality to Real GEMM or between autotune choices.
-/
/-! Scope: hA/hB pin the independent arrays to physical storage values. The
original wrapper does not resolve Torch lazy-negative flags; these premises do
not cover its logical values for such views. Proofs/LogicalViewDefect.lean and
the task report establish that source defect. No false-flag restriction is
silently added, and these raw-storage theorems earn no whole-task acceptance. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
noncomputable section

/-- Successful complete launch, all returned matrix coordinates, complete frame
and proved source bounds/unique ownership. No final-output assumption. -/
specification matmul_autotune_raw_storage_fp32_host_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (tensor : TensorContract r initial) (indices : IndexDomain r) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=Cell.mk .fp16 (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset) ∧
      SourceSafety r initial tensor := by
  obtain ⟨final,he,hv,hf⟩ := host_correct M D Q law r initial
    tensor.freshA tensor.freshB tensor.dimensions tensor.contiguousA
  refine ⟨final,he,?_,hf,source_safety r initial tensor indices⟩
  intro i j
  simpa only [reference_logical Q r initial A B hA hB i j] using hv i.val i.isLt j.val j.isLt

#axiomsClean matmul_autotune_raw_storage_fp32_host_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
