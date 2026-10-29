import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL.Expansion.Main
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.TypedStores
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Contract
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Correct
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.FPEquiv
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.IO
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Mathematical
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Misc.NativeCorrect
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Misc.TensorBridge
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.PairIO
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.AccessSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Bounds
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.CanonicalTrace
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPBody
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPExecution
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPInvariant
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPLoop
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPProgram
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPReadback
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPSetup
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPTail
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPValues
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPWorkset
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOBinding
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOConnection
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOGeometry
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOImplements
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOLocality
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOOldRun
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOPrograms
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOProjection
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOTrace
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOValues
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.KernelInterface
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LegacySurface
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LogicalReader
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.MemoryTransport
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Native
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeGraph
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeKernelResult
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairConnection
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairIO
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Physical
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PublicIO
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Real
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealFormula
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealKernelResult
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealTensor
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSourceSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.ScalarTree
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourceControl
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourcePreservation
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourceSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TargetBridge
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportBody
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportComplete
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportLoop
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportTail
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.SurfaceSource
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Target
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Types
import bench.tests.AtomicStoreDTypeBody

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`VeriTile.Triton.DSL.CallSignature,`VeriTile.Triton.DSL.Expansion.Main,`VeriTile.Triton.DSL.Syntax,`VeriTile.Triton.DSL.TypedStores,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Contract,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Correct,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.FPEquiv,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.IO,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Mathematical,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Misc.NativeCorrect,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Misc.TensorBridge,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.PairIO,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.AccessSafety,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Bounds,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.CanonicalTrace,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPBody,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPExecution,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPInvariant,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPLoop,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPProgram,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPReadback,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPSetup,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPTail,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPValues,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPWorkset,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOBinding,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOConnection,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOGeometry,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOImplements,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOLocality,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOOldRun,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOPrograms,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOProjection,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOTrace,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOValues,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.KernelInterface,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LegacySurface,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LogicalReader,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.MemoryTransport,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Native,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeGraph,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeKernelResult,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairConnection,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairIO,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Physical,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PublicIO,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Real,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealFormula,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealKernelResult,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealTensor,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSafety,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSourceSafety,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.ScalarTree,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourceControl,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourcePreservation,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.SourceSafety,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TargetBridge,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportBody,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportComplete,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportLoop,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportTail,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.SurfaceSource,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Target,`bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Types,`bench.tests.AtomicStoreDTypeBody]
  for m in modules do
    unless env.header.modules.any (·.module == m) do
      throwError "Audit root not imported: {m}"
  let mut names : Array Name := #[]
  for (name, info) in env.constants.toList do
    let .thmInfo _ := info | continue
    let some idx := env.getModuleIdxFor? name | continue
    if modules.contains env.header.modules[idx.toNat]!.module then
      names := names.push name
  names := names.qsort (·.toString < ·.toString)
  -- One immutable checked environment, with shared visited-declaration state.
  -- The accumulated dependency union is checked after every root. Therefore
  -- each individual root's axiom footprint is contained in the standard base.
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let mut footprint : CollectAxioms.State := {}
  for name in names do
    unless (env.checked.get.find? name).isSome do
      throwError "Audit root missing from checked environment: {name}"
    let (_, next) := ((CollectAxioms.collect name).run env).run footprint
    footprint := next
    let bad := footprint.axioms.filter (fun a => ! allowed.contains a)
    unless bad.isEmpty do
      throwError "DISALLOWED axiom dependency from audit prefix ending at {name}: {bad}"
    logInfo m!"{name.toString}: axiom footprint ⊆ standard base ✓ (shared traversal)"
  logInfo m!"Checked dependency union: {footprint.axioms}"
  logInfo m!"Task theorem audit count: {names.size}"
#print_fp_assumptions VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Equivalence.scalar_left_strict
