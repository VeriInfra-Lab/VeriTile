import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.multinomial_sampling.Contract
import bench.repairs.tritonbench_g.multinomial_sampling.Correct
import bench.repairs.tritonbench_g.multinomial_sampling.FPEquiv
import bench.repairs.tritonbench_g.multinomial_sampling.IO
import bench.repairs.tritonbench_g.multinomial_sampling.Kernels
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.CorrectPublic
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.FPEquivPublic
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostCorrect
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostFPEquiv
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.NativeCorrect
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.TensorBridge
import bench.repairs.tritonbench_g.multinomial_sampling.PairIO
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Bits
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Boundary
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.CDFBridge
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Contract
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBackend
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBody
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPHost
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPInvariant
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPPrologue
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSource
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPStorage
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPWrapper
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.FPCompatibility
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Guarded
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Host
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOFacts
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IORealResult
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOResult
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOSafety
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOUndef
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelCompatibility
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelMemory
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalReader
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalTransport
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Native
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.NumericTrees
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Numerics
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Physical
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Real
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.SourceSafety
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorInputs
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorReference
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Unrolled
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.UnrolledPair

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`bench.repairs.tritonbench_g.multinomial_sampling.Contract,`bench.repairs.tritonbench_g.multinomial_sampling.Correct,`bench.repairs.tritonbench_g.multinomial_sampling.FPEquiv,`bench.repairs.tritonbench_g.multinomial_sampling.IO,`bench.repairs.tritonbench_g.multinomial_sampling.Kernels,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.CorrectPublic,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.FPEquivPublic,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostCorrect,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostFPEquiv,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.NativeCorrect,`bench.repairs.tritonbench_g.multinomial_sampling.Misc.TensorBridge,`bench.repairs.tritonbench_g.multinomial_sampling.PairIO,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Bits,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Boundary,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.CDFBridge,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Contract,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBackend,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBody,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPHost,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPInvariant,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPPrologue,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSource,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPStorage,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPWrapper,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.FPCompatibility,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Guarded,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Host,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOComplete,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOFacts,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IORealResult,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOResult,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOSafety,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOUndef,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelCompatibility,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelMemory,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalReader,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalTransport,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Native,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.NumericTrees,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Numerics,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Physical,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.ReaderBefore,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Real,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.SourceSafety,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorInputs,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorReference,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Unrolled,`bench.repairs.tritonbench_g.multinomial_sampling.Proofs.UnrolledPair]
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
#print_fp_assumptions VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv.static_unroll_equiv
