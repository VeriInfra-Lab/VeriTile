import bench.tritonbench_g.fast_rope_embedding.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open scoped VeriTile.Spec
noncomputable section

/-- Actual FP32 original invocation with the existing native lowering profile. -/
def original (cfg : FP.Config) (r : Request) (QR CR SR : Nat) (backward : Bool) (row group : Nat) :=
  ProgramIO.Implementation.original (IOPair.io cfg r QR CR SR backward row group) (fun _ => True)

/-- Preserved integer head-group commutation candidate, on the same logical IO. -/
def candidate (cfg : FP.Config) (r : Request) (QR CR SR : Nat) (backward : Bool) (row group : Nat) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant cfg r QR CR SR backward row group) (fun _ => True)

/-- Both full kernels are safe and succeed on every launched row/head group;
all active typed output cells agree and both complete exterior frames hold.
Both directions, all FP32 words, masks, casts, dynamic loops and actual strides
are retained. Only a natural-number head-group product is commuted. -/
specification fast_rope_kernel_equivalent (cfg : FP.Config) (r : Request) (QR CR SR : Nat)
    (backward : Bool) (row group : Nat) (launch : r.Launched row group)
    (strideEq : row * QR = row * (r.heads * r.dimension))
    (cosineEq : CR = r.half) (sineEq : SR = r.half) :
    original cfg r QR CR SR backward row group ≡[[]]
      candidate cfg r QR CR SR backward row group := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent cfg r QR CR SR backward row group launch strideEq cosineEq sineEq

#axiomsClean fast_rope_kernel_equivalent
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.FPEquiv
