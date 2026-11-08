import bench.tritonbench_g.fast_rope_embedding.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Actual FP32 source, Q's in-place group window, and C/S coefficient rows.
IO.lean displays all masks, stride arguments, and the complete dynamic loop. -/
def ropeIO (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) :=
  LogicalIO.view r qStride cosineStride sineStride backward row group

/-- Independent two-coordinate rotation; odd trailing columns are framed. -/
def mathematical (r : Request) (backward : Bool) (values : LogicalIO.Inputs r) :
    Fin r.heads → Fin (2*r.half) → ℝ := fun h c =>
  let i := Interface.lane r c
  let a := values.query h (Interface.left r i)
  let b := values.query h (Interface.right r i)
  let cosine := values.cosine i
  let sine := values.sine i
  let pair := if backward then (a*cosine+b*sine, b*cosine-a*sine)
    else (a*cosine-b*sine, b*cosine+a*sine)
  if r.half ≤ c.val then pair.2 else pair.1

specification fast_rope_kernel_correctness (r : Request)
    (qStride cosineStride sineStride : Nat) (backward : Bool) (row group : Nat)
    (launch : r.Launched row group) (strideEq : row * qStride = row * (r.heads * r.dimension))
    (cosineEq : cosineStride = r.half) (sineEq : sineStride = r.half) :
    VeriTile.Spec.Real (ropeIO r qStride cosineStride sineStride backward row group ⊨ mathematical r backward) :=
  (LogicalIO.judgment_iff r qStride cosineStride sineStride backward row group _).mpr (Interface.real_implements r qStride cosineStride sineStride backward row group launch strideEq cosineEq sineEq)

#axiomsClean fast_rope_kernel_correctness
#specNonCircular mathematical avoiding [Kernels.rope, Source.source, FP.Structural.exec]
#stmtSurfaceSubset fast_rope_kernel_correctness ⊆
  [LogicalIO.Inputs, VeriTile.Spec.Real, ropeIO, mathematical, LogicalProgramIO.Implements, Request,
    Request.Launched, Request.heads, Request.dimension, Request.half]
end
end VeriTile.Bench.Repairs.FastRopeEmbedding
