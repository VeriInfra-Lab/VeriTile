import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.RawNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding KernelInterface
noncomputable section

/-- Complete FP32 source execution, independent native output words and every
unwritten raw memory cell. The existing primitive/lowering interpretation is retained. -/
theorem fast_rope_raw_fp32 (cfg : FP.Config) (r : Request)
    (qStride cosineStride sineStride : Nat) (backward : Bool)
    (s : State Numerics.Tagged) (row group : Nat) (launch : r.Launched row group)
    (strideEq : row*qStride=row*(r.heads*r.dimension))
    (cosineEq : cosineStride=r.half) (sineEq : sineStride=r.half) :
    ∃ final, FP.Structural.exec (Numerics.operations (Numerics.softwarePrimitives cfg)).backend
        (rawSource r qStride cosineStride sineStride backward) (programState s row group)=some final ∧
      (∀ h col,group*4≤h → h<min (group*4+4) r.heads → col<2*r.half →
        ((final.mem "Q" (r.offset row h col)).read .real).word=
          Numerics.result (Numerics.softwarePrimitives cfg) r backward
            (fun o=>((s.mem "Q" o).read .real).word)
            (fun o=>((s.mem "C" o).read .real).word)
            (fun o=>((s.mem "S" o).read .real).word) row h col) ∧
      (∀ region offset,¬Kernel.Written r row group region offset →
        final.mem region offset=s.mem region offset) := by
  obtain ⟨final,ran,post⟩ := KernelInterface.executes
    (Numerics.operations (Numerics.softwarePrimitives cfg)) r qStride cosineStride sineStride
    backward s row group launch strideEq cosineEq sineEq
  refine ⟨final,ran,?_,post.frame⟩
  intro h col hlo hhi hc
  rw [post.values h col hlo hhi hc,Cell.read_same]
  exact Numerics.source_profile _ _ _ _ _ _ _ _ _

#axiomsClean fast_rope_raw_fp32
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.RawNative
