import bench.repairs.tritonbench_g.embedding_triton_kernel.IO
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

/-- Both source variants use the same mixed metadata/payload IO and exact word
copy/positive-zero primitive model. The complete IO fields are in `IO.lean`. -/
def embeddingIO (rest : FP.Structural.Algebra (BitVec 32)) (candidate : Bool)
    (p : Parameters) (pid : Nat) : FP.Structural.ProgramIO (BitVec 32) (BitVec 32) :=
  IO.embeddingIO rest 0 candidate p pid

/-- Independent bit-preserving vocabulary lookup, with positive FP32 zero for
invalid IDs. Every FP32 word is admitted. -/
def reference (p : Parameters) (ids : IO.Ids) (weight : IO.Weights (BitVec 32) p)
    (a : Fin 64) (d : Fin p.width) : BitVec 32 :=
  if valid : p.first ≤ ids a ∧ ids a<p.last then
    weight ⟨(ids a-p.first).toNat,by unfold IO.vocabulary; omega⟩ d
  else 0

/-- For either equivalent source, actual execution is safe and successful,
with the complete independent word result and exterior typed-cell frame. -/
theorem embedding_kernel_successful_equivalence
    (rest : FP.Structural.Algebra (BitVec 32)) (candidate : Bool)
    (p : Parameters) (pid : Nat) (layout : Execution.Layout p) :
    (embeddingIO rest candidate p pid).ImplementsOn (IO.signed p pid)
      (IO.outputValues p (reference p)) :=
  IOImplements.implements_signed rest 0 candidate p pid layout

#specNonCircular reference avoiding [source,indexCommutedSource,FP.Structural.exec]
#axiomsClean embedding_kernel_successful_equivalence
#stmtSurfaceSubset embedding_kernel_successful_equivalence ⊆
  [embeddingIO,reference,Execution.Layout,IO.outputValues,IO.signed,
   Parameters,FP.Structural.Algebra,FP.Structural.ProgramIO.ImplementsOn]
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
