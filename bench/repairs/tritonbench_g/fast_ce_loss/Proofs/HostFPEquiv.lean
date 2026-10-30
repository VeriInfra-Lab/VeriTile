import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ResidentWhole
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Contract

namespace VeriTile.Bench.Repairs.FastCeLoss.FPEquiv
open Source Host Floating Resident
noncomputable section

/-!
Whole native returned-byte correspondence for the repaired FP32/int64 source.
IO: raw/resolved contiguous rank-3 logits, raw int64 labels and raw/resolved
0-D upstream; nonzero storage offsets and stored negative values are supported.
Internal stride-zero expansion and exact custom/public return aliases follow
from actual allocation/view laws. Native and marker models use the same actual
artifact, full AST, raw Python scalar payloads, calls, allocations and trace.
The fixed current reaching ID determines each plan before source-expression
transport. Every reduction expands to scalar nodes with covered lanes; the
actual selected order, default fusion, operand swaps and same-type cast fusion
remain explicit. No FMA=separate multiplication/addition law is assumed.

LocalPorts is the only numerical equality premise: fixed-arity FP32 scalar
operations on their actual native operand tuples. Tape.Admitted lists those
local domains throughout the actual trace, including conversions, libdevice
tanh, exp/log, host arithmetic and each reduction node. CompilerCorrect is an
explicit ordinary compiler/frontend/runtime boundary, not a numerical result
oracle. The theorem derives complete source executions, output bytes, custom
tuple, caller bytes and public gradient aliases from one successful native run.
No GPU execution, native numerical conformance or Real accuracy is asserted.
-/

 theorem complete {σ : Type} (engine : Engine σ) (compiler : CompilerCorrect engine)
    (runtime : Runtime σ) (library : Library runtime engine.live) (args : Arguments) (upstream : Descriptor)
    (initial final : σ) (result : WholeReturn) (artifact : engine.Artifact) (abi : CompleteABI σ) (trace : engine.Trace)
    (reference : Numerics) (domain : Domain) (ports : LocalPorts engine.native reference domain)
    (admitted : trace.tape.Admitted engine.native domain)
    (input : InputContract engine.live args upstream initial (rawInteger engine.physical initial))
    (abiArgs : abi.args=args) (abiUpstream : abi.scalarUpstream=upstream) (abiInitial : abi.initial=initial)
    (fusion : abi.enableFPFusion=Bool.true)
    (installed : engine.resident ⟨args,upstream⟩ ((language σ).key ⟨args,upstream⟩ abi) artifact)
    (numericPrimitives : engine.primitives ⟨.numeric,runtime,abi.stream⟩ artifact)
    (markerPrimitives : engine.primitives ⟨.marker,runtime,abi.stream⟩ artifact)
    (execution : engine.compiler.execute ⟨args,upstream⟩ artifact abi initial result final trace) :
    NativeSpecification engine reference trace args upstream initial final result ∧ result=abi.returned :=
  whole_native engine compiler runtime library args upstream initial final result artifact abi trace reference domain ports
    admitted input abiArgs abiUpstream abiInitial fusion installed numericPrimitives markerPrimitives execution

/-- Actual successful native body execution composes with the exact checked
guards and hook. The resulting checked source execution is derived from the
resident compiler boundary, together with the full returned native specification. -/
theorem checked_complete {σ : Type} (engine : Engine σ) (compiler : CompilerCorrect engine)
    (runtime : Runtime σ) (library : Library runtime engine.live)
    (fm : Checked.ForwardMetadata) (bm : Checked.BackwardMetadata)
    (args : Arguments) (upstream : Descriptor)
    (initial final : σ) (result : WholeReturn) (artifact : engine.Artifact) (abi : CompleteABI σ) (trace : engine.Trace)
    (reference : Numerics) (domain : Domain) (ports : LocalPorts engine.native reference domain)
    (admitted : trace.tape.Admitted engine.native domain)
    (forward : Checked.ForwardContract fm args (rawInteger engine.physical initial))
    (backward : Checked.BackwardContract bm args upstream)
    (bindings : Checked.Bindings engine.live args upstream initial)
    (abiArgs : abi.args=args) (abiUpstream : abi.scalarUpstream=upstream) (abiInitial : abi.initial=initial)
    (fusion : abi.enableFPFusion=Bool.true)
    (installed : engine.resident ⟨args,upstream⟩ ((language σ).key ⟨args,upstream⟩ abi) artifact)
    (numericPrimitives : engine.primitives ⟨.numeric,runtime,abi.stream⟩ artifact)
    (markerPrimitives : engine.primitives ⟨.marker,runtime,abi.stream⟩ artifact)
    (execution : engine.compiler.execute ⟨args,upstream⟩ artifact abi initial result final trace) :
    (∃ sourceFinal, Checked.whole Floating.arithmetic fpScalar runtime abi.stream fm bm args upstream
      (initialWorld engine initial .numeric)=.ok result sourceFinal ∧ sourceFinal.native=final) ∧
    NativeSpecification engine reference trace args upstream initial final result ∧ result=abi.returned := by
  have input := Checked.input_contract forward backward bindings
  have coherent : (language σ).coherent ⟨.numeric,runtime,abi.stream⟩ ⟨args,upstream⟩ abi
      (initialWorld engine initial .numeric) := ⟨abiArgs,abiUpstream,abiInitial,rfl,fusion,rfl,rfl⟩
  obtain ⟨sourceFinal,sourceRun,sourceState,_,_⟩ := compiler.refinement.simulate
    ⟨.numeric,runtime,abi.stream⟩ ⟨args,upstream⟩ artifact abi (initialWorld engine initial .numeric)
    result final trace installed ⟨numericPrimitives,library⟩ coherent
    (initial_represented engine initial .numeric) execution
  refine ⟨⟨sourceFinal,?_,sourceState⟩,complete engine compiler runtime library args upstream initial final result
    artifact abi trace reference domain ports admitted input abiArgs abiUpstream abiInitial fusion installed
    numericPrimitives markerPrimitives execution⟩
  rw [Checked.whole_admitted Floating.arithmetic fpScalar runtime abi.stream fm bm args upstream
    (initialWorld engine initial .numeric) forward backward]
  exact sourceRun

/-- Every FP32 input pattern, including NaNs, yields the selected literal-zero
source expression on ignored rows. There is no multiplication-by-zero premise. -/
theorem ignored_expression (options : Options) (col : Nat) (x lse upstream : Expr) :
    Source.backwardScalar Floating.arithmetic options (-100) col x lse upstream=
      .unary .cast (.literal 0) := Source.backward_ignored _ _ _ _ _ _

#axiomsClean complete
#axiomsClean checked_complete
#axiomsClean ignored_expression
#specNonCircular LocalPorts avoiding [Source.forwardGrid,Source.backwardGrid,Host.whole,Math.loss,Math.gradient]
#specNonCircular Engine avoiding [Math.loss,Math.gradient,Complete.lseGraph,Complete.scalarGraph,API.gradientGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.FPEquiv
