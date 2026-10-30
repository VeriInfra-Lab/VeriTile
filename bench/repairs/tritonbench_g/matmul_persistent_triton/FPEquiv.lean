import bench.repairs.tritonbench_g.matmul_persistent_triton.Contract
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.ResidentWhole

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.FPEquiv
open Host Floating Resident
noncomputable section

/-!
Whole returned-byte correspondence for both production storage profiles:
FP16 BK64/stages3 and E4M3 BK128/stages4, BM128/BN256/GROUP8/warps8.
The source keeps the supplied FP32 accumulator, original K-block order,
zero-filled K-tail loads, explicit output conversion and default fusion.
The input domain is ordinary raw/resolved, same-device, matching supported
storage with nonnegative strided backing; empty outputs and K=0 are included.

One installed artifact, exact complete AST/ABI and native execution supply
both numeric and byte-marker interpretations. Plans are fixed at actual
current reaching IDs and partition every source K lane exactly once.
The explicit ISA catalog contains SM90 WGMMA and fixed MMA variants;
other target instruction lowerings are not asserted by this theorem.

LocalPorts is the only numerical equality hypothesis: actual output cast or
one named fixed-size instruction on its actual operands. There is no whole
dot/reduction/output port and no FMA-versus-separate-mul/add equality.
CompilerCorrect and the two Library contracts are ordinary implementation
boundaries. The theorem derives source completion, return identity/metadata,
freshness, every output byte and every caller byte from the native run.
CPU/offline controls do not assert GPU execution, native numerical conformance
or equality between rounded native arithmetic and the independent Real target.
-/

theorem complete {σ : Type} (engine : Engine σ) (compiler : CompilerCorrect engine)
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
      abi.launches=abi.calls.filterMap callABI :=
  whole_native engine compiler runtime library physicalLibrary args initial final output artifact abi trace
    reference domain ports admitted input binding abiArgs abiInitial fusion installed numericPrimitives markerPrimitives execution


theorem checked_complete {σ : Type} (engine : Engine σ) (compiler : CompilerCorrect engine)
    (runtime : Runtime σ) (library : Host.Library runtime engine.live)
    (physicalLibrary : Physical.Library runtime engine.physical)
    (args : Arguments) (initial final : σ) (output : Tensor)
    (artifact : engine.Artifact) (abi : CompleteABI σ) (trace : engine.Trace)
    (reference : Numerics) (domain : Domain) (ports : LocalPorts engine.native reference domain)
    (admitted : trace.tape.Admitted engine.native domain)
    (guard : Checked.checkInputs args = .ok ())
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : engine.live initial args.A.storage) (bLive : engine.live initial args.B.storage) (binding : InputBinding engine.physical args initial)
    (abiArgs : abi.args=args) (abiInitial : abi.initial=initial) (fusion : abi.enableFPFusion=Bool.true)
    (installed : engine.resident args ((language σ).key args abi) artifact)
    (numericPrimitives : engine.primitives ⟨.numeric,runtime,engine.live,abi.stream⟩ artifact)
    (markerPrimitives : engine.primitives ⟨.marker,runtime,engine.live,abi.stream⟩ artifact)
    (execution : engine.compiler.execute args artifact abi initial output final trace) :
    NativeSpecification engine reference trace args initial final output ∧ output=abi.returned ∧
      abi.launches=abi.calls.filterMap callABI :=
  whole_native engine compiler runtime library physicalLibrary args initial final output artifact abi trace
    reference domain ports admitted
      (Checked.admitted_input engine.live args initial guard aBound bBound aLive bLive) binding abiArgs abiInitial fusion installed numericPrimitives markerPrimitives execution


#axiomsClean checked_complete
#axiomsClean complete
#specNonCircular LocalPorts avoiding [Floating.reference,Host.run,Source.run,NativeSpecification]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.FPEquiv
