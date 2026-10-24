import bench.repairs.tritonbench_g.kldiv_ops.Proofs.LocalNumerics
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SourceBytes
import bench.repairs.tritonbench_g.kldiv_ops.Misc.RealBackward
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.AllFrames
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.TypedPreservation

/-! Complete FP32 return-value graph for the separately named repair.
The total four-byte carrier includes NaNs, infinities, signed zeros and
subnormals. Scalar software operations use RN-even with no flushing. Native
SM80 conformance is deferred locally; no GPU execution is claimed. max/log/exp,
per-block reduction and the two Torch sums/scalar-int divide are explicit
primitive ports. The graph retains every FP32 reduction block and sequential
carry add, the original block cap/warp branches, and separate sub/mul sites.

Returned values below include every host reduction, normalized input view,
cotangent snapshot, copyback, identity branch and broadcast multiplication.
The capacity guards are actual source actions. Independent Real interpretation
is RealHost; no universal exact floating-to-Real hypothesis is made.

The two printed empty FP-atom lists concern the reused unchanged complete
backward bodies' pointer-product commutation only. Whole-wrapper returned
word equalities are proved by ForwardValues/BackwardValues, not by those lists.
-/
namespace VeriTile.Bench.Repairs.KldivOps.FPEquivalence
open Storage Host FPProfile VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance : Inhabited Word := ⟨VeriTile.Triton.FP.Value.zero .fp32⟩

def codec : Codec Word := VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec.littleEndian

/-- Exact complete FP32 software/primitive-port return, with no assumed
whole-kernel value or footprint law. -/
theorem forward (rest : Algebra Word) (sum : Bool→List Word→Word) (divide : Word→Nat→Word)
    (api : Services Word) (lib : Library codec (numerics sum divide) api) (compiler : Compiler)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State)
    (predLive : prediction.backing∈entry.live) (targetLive : target.backing∈entry.live)
    (call : Host.forward api (sourceActions codec (algebra rest) compiler) prediction target flag name eps entry=.ok returned final) :
    ∃ f : ForwardResult codec (numerics sum divide) (algebra rest) api compiler prediction target flag name eps entry returned final,
      ∀ i,i<returned.numel → value codec (numerics sum divide).neg final.heap returned i=
        reducedWord (numerics sum divide) f.trace.red f.trace.rows f.trace.cols
          (forwardKernelValue (algebra rest) f.trace.red f.trace.cols flag eps
            (value codec (numerics sum divide).neg entry.heap prediction)
            (value codec (numerics sum divide).neg entry.heap target)) i := by
  obtain ⟨f⟩ := forward_correct codec (numerics sum divide) (algebra rest) api lib compiler
    prediction target flag name eps entry returned final predLive targetLive call
  exact ⟨f,f.returned_values f.capacity⟩

theorem backward (rest : Algebra Word) (sum : Bool→List Word→Word) (divide : Word→Nat→Word)
    (api : Services Word) (lib : Library codec (numerics sum divide) api) (compiler : Compiler)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State)
    (tlive : target.backing∈entry.live) (glive : grad.backing∈entry.live) (dlive : dst.backing∈entry.live)
    (call : Host.backward api (sourceActions codec (algebra rest) compiler) target grad dst flag entry=.ok returned final) :
    ∃ f : BackwardResult codec (numerics sum divide) (algebra rest) api compiler target grad dst flag entry returned final,
      ∀ i,i<returned.numel → value codec (numerics sum divide).neg final.heap returned i=
        if f.finish.isOne then backwardKernelValue (algebra rest) flag target.numel
          (value codec (numerics sum divide).neg entry.heap target)
          (value codec (numerics sum divide).neg f.trace.ready.heap f.trace.work) i
        else (numerics sum divide).multiply
          (backwardKernelValue (algebra rest) flag target.numel
            (value codec (numerics sum divide).neg entry.heap target)
            (value codec (numerics sum divide).neg f.trace.ready.heap f.trace.work)
            (broadcastIndex returned.shape dst.shape i))
          (value codec (numerics sum divide).neg entry.heap grad (broadcastIndex returned.shape grad.shape i)) := by
  obtain ⟨f⟩ := backward_correct codec (numerics sum divide) (algebra rest) api lib compiler
    target grad dst flag entry returned final tlive glive dlive call
  exact ⟨f,f.returned_values f.capacity⟩

abbrev defaultLocalError := @LocalNumerics.default_error
abbrev logLocalError := @LocalNumerics.log_error
abbrev unaryLocalBound := @LocalNumerics.UnaryBound
abbrev binaryLocalBound := @LocalNumerics.BinaryBound
abbrev defaultNumericalDomain := @LocalNumerics.DefaultDomain
abbrev logNumericalDomain := @LocalNumerics.LogDomain
abbrev totalForwardGraph := @ForwardResult.returned_values
abbrev totalBackwardGraph := @BackwardResult.returned_values
abbrev defaultScalar := @scalar_default
abbrev logScalar := @scalar_log
abbrev idealForward := @RealExecution.returned_forward_ideal
abbrev idealBackward := @RealExecution.returned_backward_ideal
abbrev allForwardFrames := @forward_all_frames
abbrev allBackwardFrames := @backward_all_frames
abbrev actualBackwardStores := @BackwardResult.source_bytes
abbrev defaultBodyPreservation := @TypedPreservation.default_body
abbrev logBodyPreservation := @TypedPreservation.log_body
#print_fp_assumptions defaultBodyPreservation
#print_fp_assumptions logBodyPreservation

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.FPEquivalence
