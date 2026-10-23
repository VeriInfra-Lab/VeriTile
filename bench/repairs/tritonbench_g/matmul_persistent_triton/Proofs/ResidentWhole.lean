import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.ResidentAdapter

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Resident
open Host Floating
noncomputable section
set_option maxHeartbeats 7000000

structure InputBinding {σ : Type} (physical : Physical.Layout σ) (args : Arguments) (initial : σ) : Prop where
  aUnit : physical.unit args.A.storage=args.A.dtype.bytes
  bUnit : physical.unit args.B.storage=args.B.dtype.bytes
  aCapacity : physical.capacity initial args.A.storage=args.A.allocationBytes
  bCapacity : physical.capacity initial args.B.storage=args.B.allocationBytes

def StoredReference {σ : Type} (engine : Engine σ) (reference : Numerics) (trace : engine.Trace)
    (final : σ) (output : Tensor) (row column : Nat) (expression : Expr) : Prop :=
  ∃ id node,trace.tape.final.cells (output.address row column)=some id ∧
    trace.tape.final.nodes id=some node ∧ ∃ same : node.expression=expression,
    output.dtype.bytes*((output.address row column).2+1)≤engine.physical.capacity final output.storage ∧
    ∀ lane<output.dtype.bytes,
      engine.physical.observe final output.storage (output.dtype.bytes*(output.address row column).2+lane)=
        Physical.wordByte ((node.transport same).graph.eval reference) lane

structure NativeSpecification {σ : Type} (engine : Engine σ) (reference : Numerics) (trace : engine.Trace)
    (args : Arguments) (initial final : σ) (output : Tensor) : Prop where
  values : ∀ row<args.A.rows,∀ column<args.B.columns,
    StoredReference engine reference trace final output row column
      (Floating.reference args (initialWorld engine initial .numeric).memory row column)
  callerBytes : Physical.CallerFrame engine.physical engine.live initial final
  rows : output.rows=args.A.rows
  columns : output.columns=args.B.columns
  dtype : output.dtype=args.A.dtype
  device : output.device=args.A.device
  offset : output.offset=0
  rowStride : output.rowStride=max args.B.columns 1
  columnStride : output.columnStride=1
  raw : output.isNeg=Bool.false ∧ output.isConj=Bool.false
  bounded : output.Bounded
  fresh : ∀ storage,engine.live initial storage → output.storage≠storage
  inputUnits : engine.physical.unit args.A.storage=args.A.dtype.bytes ∧
    engine.physical.unit args.B.storage=args.B.dtype.bytes

theorem whole_native {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (runtime : Runtime σ) (library : Host.Library runtime engine.live)
    (physicalLibrary : Physical.Library runtime engine.physical)
    (args : Arguments) (initial final : σ) (output : Tensor)
    (artifact : engine.Artifact) (abi : CompleteABI σ) (trace : engine.Trace)
    (reference : Numerics) (domain : Domain) (ports : LocalPorts engine.native reference domain)
    (admitted : trace.tape.Admitted engine.native domain)
    (input : InputContract engine.live args initial) (binding : InputBinding engine.physical args initial)
    (abiArgs : abi.args=args) (abiInitial : abi.initial=initial) (fusion : abi.enableFPFusion=Bool.true)
    (installed : engine.resident args ((language σ).key args abi) artifact)
    (numericPrimitives : engine.primitives ⟨.numeric,runtime,engine.live,abi.stream⟩ artifact)
    (markerPrimitives : engine.primitives ⟨.marker,runtime,engine.live,abi.stream⟩ artifact)
    (execution : engine.compiler.execute args artifact abi initial output final trace) :
    NativeSpecification engine reference trace args initial final output ∧ output=abi.returned ∧
      abi.launches=abi.calls.filterMap callABI := by
  let numeric : Model σ := ⟨.numeric,runtime,engine.live,abi.stream⟩
  let marker : Model σ := ⟨.marker,runtime,engine.live,abi.stream⟩
  have numericCoherent : (language σ).coherent numeric args abi (initialWorld engine initial .numeric) :=
    ⟨abiArgs,abiInitial,rfl,fusion,rfl,input⟩
  have markerCoherent : (language σ).coherent marker args abi (initialWorld engine initial .marker) :=
    ⟨abiArgs,abiInitial,rfl,fusion,rfl,input⟩
  obtain ⟨numericFinal,numericRun,numericState,numericMetadata,numericRepresentation⟩ := correct.refinement.simulate
    numeric args artifact abi (initialWorld engine initial .numeric) output final trace installed
    ⟨numericPrimitives,rfl,library,physicalLibrary⟩ numericCoherent (initial_represented engine initial .numeric) execution
  obtain ⟨markerFinal,markerRun,markerState,markerMetadata,markerRepresentation⟩ := correct.refinement.simulate
    marker args artifact abi (initialWorld engine initial .marker) output final trace installed
    ⟨markerPrimitives,rfl,library,physicalLibrary⟩ markerCoherent (initial_represented engine initial .marker) execution
  change numericFinal.native=final at numericState
  change markerFinal.native=final at markerState
  obtain ⟨numericSource⟩ := Host.complete Floating.arithmetic runtime engine.live args
    (initialWorld engine initial .numeric) numericFinal output library input (run_success numeric args _ _ _ numericRun)
  obtain ⟨markerSource⟩ := Host.complete Physical.arithmetic runtime engine.live args
    (initialWorld engine initial .marker) markerFinal output library input (run_success marker args _ _ _ markerRun)
  obtain ⟨_,_,numericRep⟩ := numericRepresentation
  obtain ⟨_,_,markerRep⟩ := markerRepresentation
  have numericRep : NumericRepresentation engine.physical engine.native trace.tape numericFinal := numericRep
  have markerRep : Physical.Representation engine.physical initial markerFinal.native markerFinal.memory := markerRep
  have allocation := library.allocation _ _ _ _ _ _ _ numericSource.allocation
  have unit := physicalLibrary.allocation _ _ _ _ _ _ _ numericSource.allocation
  have returnedUnit : engine.physical.unit output.storage=output.dtype.bytes := unit.trans (congrArg Storage.bytes allocation.dtype.symm)
  refine ⟨⟨?_,?_,allocation.rowsExact,allocation.columnsExact,allocation.dtype,allocation.deviceExact,
    allocation.offset,allocation.rowStride,allocation.columnStride,allocation.raw,allocation.bounded,
    allocation.fresh,⟨binding.aUnit,binding.bUnit⟩⟩,?_,?_⟩
  · intro row hr column hc
    have stored := numericRep.reference domain ports admitted _ _
      (Floating.Completion.output_reference numericSource row column hr hc)
      (reference_computed args (initialWorld engine initial .numeric).memory row column)
    simpa only [StoredReference,Tensor.address,returnedUnit,numericState] using stored
  · have bytes := Physical.caller_bytes engine.physical runtime engine.live args initial markerFinal output markerSource markerRep
    simpa only [markerState] using bytes
  · exact congrArg Metadata.returned numericMetadata
  · have calls := congrArg Metadata.calls numericMetadata
    have launches := congrArg Metadata.launches numericMetadata
    change numericFinal.calls=abi.calls at calls
    change numericFinal.calls.filterMap callABI=abi.launches at launches
    rw [calls] at launches
    exact launches.symm

#axiomsClean whole_native
#specNonCircular LocalPorts avoiding [NativeSpecification,Host.run,Source.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Resident
