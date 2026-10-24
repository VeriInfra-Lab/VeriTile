import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.FusedPair
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.FPEquiv
open VeriTile Triton FP.Structural Kernel Kernel.Native
open scoped VeriTile.Spec
noncomputable section

abbrev FPWord := VeriTile.Triton.FP.Value .fp32

def originalForward (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile) :=
  ProgramIO.Implementation.original (PublicIO.forwardIO a p s tile) (fun _=>True)

def fusedForward (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile) :=
  ProgramIO.Implementation.ofVariant (Fused.forwardVariant a p s tile) (fun _=>True)

def originalBackward (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile) :=
  ProgramIO.Implementation.original (PublicIO.backwardIO a p s tile) (fun _=>True)

def fusedBackward (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile) :=
  ProgramIO.Implementation.ofVariant (Fused.backwardVariant a p s tile) (fun _=>True)

/-- Complete forward source, with the ordered accumulator multiply and add
in one assignment. No floating primitive, conversion or reduction changes. -/
specification forward_fused_equiv (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    originalForward a p s tile ≡[[]] fusedForward a p s tile :=
  ProgramIO.Implementation.of_equivalent _ (Fused.forward_equivalent a p s tile valid)

/-- Complete DQ/barrier/DKV source. Only DQ's adjacent accumulator assignments
are fused; all three output shards and both exterior frames are preserved. -/
specification backward_fused_equiv (a : Scalar FPWord) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) :
    originalBackward a p s tile ≡[[]] fusedBackward a p s tile :=
  ProgramIO.Implementation.of_equivalent _ (Fused.backward_equivalent a p s tile valid)

#print_fp_assumptions forward_fused_equiv
#print_fp_assumptions backward_fused_equiv
#axiomsClean forward_fused_equiv
#axiomsClean backward_fused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.FPEquiv
