import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.ResidentWhole

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.FPEquiv
open Host Floating Resident
noncomputable section

/-!
Whole returned-byte correspondence for selected FP32 A/B and fixed FP16 output,
BM128/BN256/BK64/GROUP8/stages2/warps8, with both activation branches.
The source keeps its supplied FP32 accumulator, original K-block order,
zero-filled K tails, left-side slope multiplication, final cast and default fusion.
Ordinary raw/resolved A is contiguous; B may have nonnegative strides, offsets
and read-only aliasing. Empty outputs, K=0 and finite autotune replay are included.

One installed artifact, exact complete AST/ABI and native execution supply
numeric and byte-marker interpretations. Plans use actual current reaching IDs
and partition all 64 source K lanes into eight covered K8 fragments. The native
catalog is the recorded SM90 WGMMA m64n256k8 FP32/TF32 instruction only.
No separate scalar TF32 conversion or unobserved target mapping is asserted.

LocalPorts has four numerical equality fields: FP32-to-FP16 cast, comparison
against +0, one slope multiplication with its selected native operand order,
and one fixed WGMMA instruction. The source reference keeps 0.01*x even when
the native instruction swaps operands. Tape.Admitted locates the actual operands.
There is no whole-dot, reduction, activation or output numerical equality port.

CompilerCorrect and Library are ordinary implementation boundaries for the
custom complete source interpreter, exact AST and ABI. Source completion,
returned metadata, freshness, all output bytes and caller bytes are derived.
There is no generic typed-DSL translation, actual JIT binder or GPU conformance
claim, and no equality of rounded native arithmetic with the independent Real target.
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

#axiomsClean complete
#specNonCircular LocalPorts avoiding [Floating.reference,Host.run,Source.run,NativeSpecification]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.FPEquiv
