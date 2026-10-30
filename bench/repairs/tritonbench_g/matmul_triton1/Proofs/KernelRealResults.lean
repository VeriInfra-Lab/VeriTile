import bench.repairs.tritonbench_g.matmul_triton1.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.MatmulTriton1.KernelRealResults
open VeriTile.Triton FP.Structural
open Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Complete finite/shared kernel execution and the independent full-K Real sum. -/
theorem matmul_triton1_real (rest : Algebra ℝ) (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input ≠ p.output) (hb : p.weight ≠ p.output) (initial : State ℝ) :
    ∃ finite typed, FiniteSource.grid Core.realPrimitives p initial = some finite ∧
      Core.grid rest Core.realPrimitives p initial = some typed ∧ finite.mem = typed.mem ∧
      RawKernel.Post p initial finite (fun i => Core.realMatmul p initial i.1.val i.2.val) ∧
      RawKernel.Post p initial typed (fun i => Core.realMatmul p initial i.1.val i.2.val) := by
  obtain ⟨finite,typed,hf,ht,hm,hp,hq⟩ :=
    RawKernel.execute rest Core.realPrimitives p canonical ready storage geometry ha hb initial
  have hBK : 0 < p.blockInner := Nat.lt_trans (by decide) ready.blockInner.1
  have convert (final : State ℝ) (h : RawKernel.Post p initial final
      (fun i => Core.logical Core.realPrimitives p initial i.1.val i.2.val)) :
      RawKernel.Post p initial final (fun i => Core.realMatmul p initial i.1.val i.2.val) := by
    refine ⟨?_,h.2⟩
    intro i hi
    simpa only [Core.logical_realMatmul p initial i.1.val i.2.val hBK] using h.1 i hi
  exact ⟨finite,typed,hf,ht,hm,convert finite hp,convert typed hq⟩

#axiomsClean matmul_triton1_real
#specNonCircular Core.realMatmul avoiding [Core.source,Core.grid,FiniteSource.grid]
end
end VeriTile.Bench.Repairs.MatmulTriton1.KernelRealResults
