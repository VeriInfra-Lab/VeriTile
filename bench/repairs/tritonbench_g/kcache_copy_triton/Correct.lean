import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.Correct
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section


/-- One complete (token,head,split) tile, retaining signed metadata routing and output injectivity. -/
specification kcache_copy_triton_real (p : Source.Parameters) (ids : Nat → Nat) :
    Spec.Real (CopyIO.logicalIO CopyIO.realBackend Bool.false p ids ⊨{CopyIO.logicalOutputInjective p ids}
      fun xs lane => xs.payload lane) :=
  (LogicalReader.full_judgment_iff p ids).mpr (LogicalReader.kcache_copy_triton_real p ids)
#axiomsClean kcache_copy_triton_real

end
end VeriTile.Bench.Repairs.KcacheCopyTriton.Correct
