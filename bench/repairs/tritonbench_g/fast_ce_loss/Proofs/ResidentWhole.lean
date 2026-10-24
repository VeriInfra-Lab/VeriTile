import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ResidentAdapter
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.GraphInputs

namespace VeriTile.Bench.Repairs.FastCeLoss.Resident
open Source Host Floating
noncomputable section
set_option maxHeartbeats 7000000

/-- Full physical bytes of the existing execution's current stored origin.
The same plan is transported along the proved source-expression equality. -/
def StoredReference {σ : Type} (engine : Engine σ) (reference : Numerics) (trace : engine.Trace)
    (final : σ) (cell : Address) (expression : Expr) : Prop :=
  ∃ id node,trace.tape.final.cells cell=some id ∧ trace.tape.final.nodes id=some node ∧
    ∃ same : node.expression=expression,
      4*(cell.2+1)≤engine.physical.capacity final cell.1 ∧
      ∀ lane : Fin 4,engine.physical.observe final cell.1 (4*cell.2+lane.val)=
        wordByte ((node.transport same).graph.eval reference) lane

def CallerFrame {σ : Type} (engine : Engine σ) (args : Arguments) (initial final : σ) : Prop :=
  ∀ storage,engine.live initial storage → ∀ byte,byte<engine.physical.capacity initial storage →
    (storage≠args.logits.tensor.storage ∨ byte<4*args.logits.tensor.offset ∨
      4*(args.logits.tensor.offset+args.shape.rows*args.shape.vocab)≤byte) →
    byte<engine.physical.capacity final storage ∧
      engine.physical.observe final storage byte=engine.physical.observe initial storage byte

structure NativeSpecification {σ : Type} (engine : Engine σ) (reference : Numerics) (trace : engine.Trace)
    (args : Arguments) (upstream : Descriptor) (initial final : σ) (result : WholeReturn) : Prop where
  scalar : StoredReference engine reference trace final (addr result.forward.scalar.tensor 0)
    (entryScalar Floating.arithmetic fpScalar args (initialWorld engine initial .numeric))
  gradient : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
    StoredReference engine reference trace final (addr args.logits.tensor (row*args.shape.vocab+col))
      (entryGradient Floating.arithmetic fpScalar args upstream (initialWorld engine initial .numeric) row col)
  callerFrame : CallerFrame engine args initial final
  customTuple : result.customBackward=[some result.forward.context.logits.tensor,none,none,none]
  gradientStorage : result.publicGradient.tensor.storage=args.logits.tensor.storage
  gradientOffset : result.publicGradient.tensor.offset=args.logits.tensor.offset
  scalarShape : result.forward.scalar.tensor.shape=[]
  gradientShape : result.publicGradient.tensor.shape=[args.batch,args.sequence,args.vocab]

 theorem marker_byte_frame {σ : Type} (engine : Engine σ) (runtime : Runtime σ) (stream : Nat)
    (args : Arguments) (upstream : Descriptor) (initial : σ) (final : World Physical.Marker σ) (result : WholeReturn)
    (e : WholeExecution Physical.markerArithmetic ⟨fun _ => .changed,fun _ => .changed⟩ runtime engine.live
      stream args upstream (initialWorld engine initial .marker) final result)
    (rep : Physical.Representation engine.physical initial final.native final.memory) :
    CallerFrame engine args initial final.native := by
  intro storage live byte bound outside
  have lane : byte%4<4 := Nat.mod_lt _ (by decide)
  have coordinate : 4*(byte/4)+byte%4=byte := by omega
  have sourceOutside : storage≠args.logits.tensor.storage ∨ byte/4<args.logits.tensor.offset ∨
      args.logits.tensor.offset+args.shape.rows*args.shape.vocab≤byte/4 := by
    rcases outside with other|beforeView|afterView
    · exact Or.inl other
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))
  have preserved := e.caller_frame (storage,byte/4) live sourceOutside
  have marked : final.memory (storage,byte/4)=.old (storage,byte/4) := preserved
  have bytes := rep (storage,byte/4) (storage,byte/4) marked (byte%4) lane (by simpa only [coordinate] using bound)
  simpa only [coordinate] using bytes

/-- One successful native execution supplies both fixed interpretations through
the ordinary compiler boundary. Complete source executions, temporary freshness,
layout, saved LSE, mean upstream, return tuple and all caller frames are derived.
No successful source run or numerical whole-result/frame premise is required. -/
theorem whole_native {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (runtime : Runtime σ) (laws : Library runtime engine.live) (args : Arguments) (upstream : Descriptor)
    (initial final : σ) (result : WholeReturn) (artifact : engine.Artifact) (abi : CompleteABI σ) (trace : engine.Trace)
    (reference : Numerics) (domain : Domain) (ports : LocalPorts engine.native reference domain)
    (admitted : trace.tape.Admitted engine.native domain)
    (inputs : InputContract engine.live args upstream initial (rawInteger engine.physical initial))
    (abiArgs : abi.args=args) (abiUpstream : abi.scalarUpstream=upstream) (abiInitial : abi.initial=initial)
    (fusion : abi.enableFPFusion=Bool.true)
    (installed : engine.resident ⟨args,upstream⟩ ((language σ).key ⟨args,upstream⟩ abi) artifact)
    (numericPrimitives : engine.primitives ⟨.numeric,runtime,abi.stream⟩ artifact)
    (markerPrimitives : engine.primitives ⟨.marker,runtime,abi.stream⟩ artifact)
    (execution : engine.compiler.execute ⟨args,upstream⟩ artifact abi initial result final trace) :
    NativeSpecification engine reference trace args upstream initial final result ∧ result=abi.returned := by
  let program : Program := ⟨args,upstream⟩
  let numeric : Model σ := ⟨.numeric,runtime,abi.stream⟩
  let marker : Model σ := ⟨.marker,runtime,abi.stream⟩
  have numericCoherent : (language σ).coherent numeric program abi (initialWorld engine initial .numeric) :=
    ⟨abiArgs,abiUpstream,abiInitial,rfl,fusion,rfl,rfl⟩
  have markerCoherent : (language σ).coherent marker program abi (initialWorld engine initial .marker) :=
    ⟨abiArgs,abiUpstream,abiInitial,rfl,fusion,rfl,rfl⟩
  obtain ⟨numericFinal,numericRun,numericState,numericMetadata,numericRepresentation⟩ := correct.refinement.simulate
    numeric program artifact abi (initialWorld engine initial .numeric) result final trace installed
    ⟨numericPrimitives,laws⟩ numericCoherent (initial_represented engine initial .numeric) execution
  obtain ⟨markerFinal,markerRun,markerState,markerMetadata,markerRepresentation⟩ := correct.refinement.simulate
    marker program artifact abi (initialWorld engine initial .marker) result final trace installed
    ⟨markerPrimitives,laws⟩ markerCoherent (initial_represented engine initial .marker) execution
  change numericFinal.native=final at numericState
  change markerFinal.native=final at markerState
  obtain ⟨numericSource⟩ := complete_execution Floating.arithmetic fpScalar runtime engine.live abi.stream args upstream
    (initialWorld engine initial .numeric) numericFinal result laws inputs.positive inputs.logitsLive inputs.labelsLive numericRun
  obtain ⟨markerSource⟩ := complete_execution Physical.markerArithmetic ⟨fun _ => .changed,fun _ => .changed⟩
    runtime engine.live abi.stream args upstream (initialWorld engine initial .marker) markerFinal result
    laws inputs.positive inputs.logitsLive inputs.labelsLive markerRun
  obtain ⟨_,_,numericRep⟩ := numericRepresentation
  obtain ⟨_,_,markerRep⟩ := markerRepresentation
  have numericRep : NumericRepresentation engine.physical engine.native trace.tape numericFinal := numericRep
  have markerRep : Physical.Representation engine.physical initial markerFinal.native markerFinal.memory := markerRep
  refine ⟨⟨?_,?_,?_,?_,numericSource.returned_alias.2.1,numericSource.returned_alias.2.2,
    (numericSource.returned_shapes laws).1,(numericSource.returned_shapes laws).2⟩,?_⟩
  · have scalar := numericRep.reference domain ports admitted _ _ numericSource.scalar_entry (scalar_computed _ _)
    simpa only [StoredReference,numericState] using scalar
  · intro row hr col hc
    have gradient := numericRep.reference domain ports admitted _ _
      (numericSource.gradient_entry inputs.upstreamLive row col hr hc) (gradient_computed _ _ _ row col)
    simpa only [StoredReference,numericState] using gradient
  · have frame := marker_byte_frame engine runtime abi.stream args upstream initial markerFinal result markerSource markerRep
    simpa only [markerState] using frame
  · rw [numericSource.backward.resultExact]
  · exact congrArg Metadata.returned numericMetadata

#axiomsClean marker_byte_frame
#axiomsClean whole_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Resident
