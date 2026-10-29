import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Interface
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Native

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.LogsumexpFwd.FPEquiv
open VeriTile.Triton FP.Structural Native
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- Complete source grid with its FP32 runtime scale and the selected native
maximum/sum trees for each program. All raw binary32 words are admitted. -/
def logsumexpIO (ops : Ops) (plans : Nat → Plans) (a : RawKernel.Geometry)
    (scale : Word) (jobs : List Nat) : ProgramIO Word Word :=
  Interface.programIO (fun j => engine ops (plans j)) a scale jobs

/-- Independent ordered native block graph. The existing approximate exp/log
primitive boundaries and reduction plans are retained without a rewrite law. -/
def native (ops : Ops) (plans : Nat → Plans) (a : RawKernel.Geometry)
    (scale : Word) (values : Interface.Inputs Word a) : Interface.Outputs Word a :=
  Interface.result (fun j => blockValue (engine ops (plans j.val)) (Interface.extend values)
    scale (j.val / a.blocks) (j.val % a.blocks) a.columns a.block a.hasScale)

theorem logsumexp_native_correctness (ops : Ops) (plans : Nat → Plans)
    (a : RawKernel.Geometry) (scale : Word) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows * a.blocks))) :
    logsumexpIO ops plans a scale jobs ⊨ native ops plans a scale :=
  Interface.implements (fun j => engine ops (plans j)) a scale jobs schedule

#axiomsClean logsumexp_native_correctness
#specNonCircular native avoiding [rowBlock, Source.rowValue, RawKernel.run]
#stmtSurfaceSubset logsumexp_native_correctness ⊆
  [logsumexpIO, ProgramIO.Implements, native, Ops, Plans, Word,
    RawKernel.Geometry, RawKernel.Geometry.rows, RawKernel.Geometry.blocks]
end
end VeriTile.Bench.Repairs.LogsumexpFwd.FPEquiv
