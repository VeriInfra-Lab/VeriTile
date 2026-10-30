import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.FusedGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.NativeIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.FPEquiv
open VeriTile Triton FP.Structural
open Whole002 Source Whole Native Checked
open scoped VeriTile.Spec
noncomputable section

def original (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (work : List Program) :=
  ArrayProgramIO.Implementation.original (NativeIO.io rt I c work) (KernelIO.domain c)

def fusedMask (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config) (work : List Program) :=
  ArrayProgramIO.Implementation.ofVariant
    (FusedMask.variant (algebra rt) (dotModel c.storage I) c work) (KernelIO.domain c)

/-- Complete sparse-block and sparse-column loops, early return, every grid
program, typed outputs and both exterior frames. The sole optimization fuses
an FP32 zero tile assignment with its immediately following masked selection;
all casts, floating operations and reduction/dot order remain unchanged. -/
specification fused_mask_equiv (rt : Runtime) (I : CompareISA (BitVec 32)) (c : Config)
    (sep : Separated c) (layout : Contiguous c) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c))) :
    original rt I c work ≡[[]] fusedMask rt I c work :=
  ArrayProgramIO.Implementation.of_equivalentOn _
    (FusedMask.equivalent (algebra rt) (dotModel c.storage I) c sep layout hbm hbn work order)

#print_fp_assumptions fused_mask_equiv
#axiomsClean fused_mask_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MixedSparseAttention.FPEquiv
