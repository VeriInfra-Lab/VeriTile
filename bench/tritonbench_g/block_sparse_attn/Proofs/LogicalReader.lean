import bench.tritonbench_g.block_sparse_attn.Spec
import bench.tritonbench_g.block_sparse_attn.Proofs.PublicCorrectness
import bench.tritonbench_g.block_sparse_attn.IO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem logical_iff (p : Parameters) (batches visits : Nat) :
    (logicalIO p batches visits ⊨{logicalDomain p batches visits} fun xs c =>
      BlockSparseAttn.Spec.output xs.layout.shape xs.layout.csr xs.tensors p.scale
        c.work.batch c.work.head c.work.row c.work.col) ↔
    (realIO p batches visits).ImplementsOn (Admissible p batches visits) (realFormula p batches visits) := by
  change (realIO p batches visits).ImplementsOn
    (fun vals => logicalDomain p batches visits (logicalInputsEquiv p batches visits vals))
    (fun vals => (logicalOutputsEquiv p batches).symm (fun c =>
      BlockSparseAttn.Spec.output (logicalInputsEquiv p batches visits vals).layout.shape
        (logicalInputsEquiv p batches visits vals).layout.csr (logicalInputsEquiv p batches visits vals).tensors
        p.scale c.work.batch c.work.head c.work.row c.work.col)) ↔ _
  have deq : (fun vals => logicalDomain p batches visits (logicalInputsEquiv p batches visits vals)) =
      Admissible p batches visits := by
    funext vals
    simp only [logicalDomain,Equiv.symm_apply_apply]
  have eq : (fun vals => (logicalOutputsEquiv p batches).symm (fun c =>
      BlockSparseAttn.Spec.output (logicalInputsEquiv p batches visits vals).layout.shape
        (logicalInputsEquiv p batches visits vals).layout.csr (logicalInputsEquiv p batches visits vals).tensors
        p.scale c.work.batch c.work.head c.work.row c.work.col)) = realFormula p batches visits := by
    funext vals o
    fin_cases o
    funext j
    simp only [logicalOutputsEquiv,Equiv.coe_fn_symm_mk,Coordinate.work,LogicalInputs.layout,
      LogicalInputs.tensors,Equiv.symm_apply_apply]
    rfl
  exact (congrArg₂ (realIO p batches visits).ImplementsOn deq eq).to_iff

theorem logical_implements (p : Parameters) (batches visits : Nat) (geometry : Geometry p) :
    logicalIO p batches visits ⊨{logicalDomain p batches visits} fun xs c =>
      BlockSparseAttn.Spec.output xs.layout.shape xs.layout.csr xs.tensors p.scale
        c.work.batch c.work.head c.work.row c.work.col :=
  (logical_iff p batches visits).mpr (real_implements p batches visits geometry)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
