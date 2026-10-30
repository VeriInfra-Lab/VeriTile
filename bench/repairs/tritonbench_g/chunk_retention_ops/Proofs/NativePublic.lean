import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Numeric
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Physical
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Public

/-- The numerical boundary consists only of eight fixed-arity scalar laws.
Default K^-0.5 and explicit user scale take separate host paths. -/
theorem host_native (c : Shape) (_abi : HostABI.LaunchABI c)
    (models : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws models ports) (tree : SumTree) (mode : Numeric.ScaleMode)
    (rf ri : Bool) (entry : Heap Numeric.Word) :
    Numeric.nativeRun ports c tree mode rf ri entry=
      Numeric.referenceRun models c tree mode rf ri entry :=
  Numeric.native_eq models ports laws c tree mode rf ri entry

/-- Exact physical-memory composition, independent of scalar numerical laws. -/
theorem host_typed (ops : Ops α) (c : Shape) (_abi : HostABI.LaunchABI c)
    (binding : Binding c) (scale : α) (tree : SumTree) (rf ri : Bool) (entry : Memory α) :
    binding.view ops.zero (Source.runPhysicalAPI ops c binding scale tree rf ri entry)=
      Source.runAPI ops c scale tree rf ri (binding.view ops.zero entry) :=
  Source.run_physical ops c binding scale tree rf ri entry

/-- Raw FP32 caller words outside the complete active-store footprint survive
unchanged, regardless of NaNs, infinities or numerical conformance. -/
theorem host_native_frame (c : Shape) (_abi : HostABI.LaunchABI c)
    (binding : Binding c) (ports : Numeric.ScalarPorts) (scale : Numeric.Word)
    (tree : SumTree) (rf ri : Bool) (entry : Memory Numeric.Word) (point : Nat)
    (outside : ∀ a∈Source.footprint c rf ++ if ri then Source.eventsInitial c else [],
      point≠binding.address a) :
    Source.runPhysicalAPI (wordOps (Numeric.nativeISA ports)) c binding scale tree rf ri entry point=entry point :=
  Source.run_physical_frame _ _ _ _ _ _ _ _ _ outside

#axiomsClean host_native
#axiomsClean host_typed
#axiomsClean host_native_frame
end Public
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

