import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelBoundary
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelTensor
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelFMA

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelPublic
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelContract KernelContract.Storage
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

namespace Real
 theorem forward (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p 1)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (warps : Nat) (geometry : KernelLaunch.ForwardGeometry p warps)
    (launch : Launch p 1) (pids : LaunchState launch words) :
    KernelBoundary.ForwardAddresses p views alloc bytes words inputs launch ∧
    let initial := KernelTensor.realState p interpret words
    let x := fun col => KernelTensor.logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.forward p) initial=some final ∧
      tensorValue final views.lse launch.row=some (KernelMath.fullLSE p x) ∧
      tensorValue final views.loss launch.row=some (KernelMath.rowLoss p x (tensorLabel words views.labels launch.row)) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row) → final.mem region offset=initial.mem region offset) := by
  constructor
  · exact KernelBoundary.forward_addresses p views alloc bytes words inputs launch
  · exact KernelTensor.forward interpret p views alloc bytes words inputs (KernelLaunch.forward_covered p geometry) launch pids

 theorem chunked (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p p.chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (warps : Nat) (geometry : KernelLaunch.ChunkedGeometry p warps)
    (launch : Launch p p.chunks) (pids : LaunchState launch words) :
    KernelBoundary.ForwardAddresses p views alloc bytes words inputs launch ∧
    let initial := KernelTensor.realState p interpret words
    let x := fun col => KernelTensor.logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.chunked p) initial=some final ∧
      tensorValue final views.lse (launch.row*p.chunks+launch.block)=some (KernelMath.blockLSE p launch.block x) ∧
      (launch.block=0 → tensorValue final views.loss launch.row=
        some (KernelMath.partialLoss p x (tensorLabel words views.labels launch.row))) ∧
      (∀ region offset, (launch.block=0 → region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row*p.chunks+launch.block) →
        final.mem region offset=initial.mem region offset) := by
  constructor
  · exact KernelBoundary.forward_addresses p views alloc bytes words inputs launch
  · have count : p.chunks=KernelMath.CE.blocks p.vocab p.block := by
      rw [geometry.width]
      exact geometry.count
    exact KernelTensor.chunked interpret p views alloc bytes words inputs count launch pids

 theorem backward (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (warps : Nat) (geometry : KernelLaunch.BackwardGeometry p blocks warps)
    (launch : Launch p blocks) (pids : LaunchState launch words) :
    KernelBoundary.BackwardAddresses p views alloc bytes words inputs launch ∧
    let initial := KernelTensor.realState p interpret words
    let x := fun col => KernelTensor.logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
        tensorValue final views.logits (launch.row*p.vocab+(launch.block*p.block+lane.val))=
          some (KernelReal.gradient p (tensorLabel words views.labels launch.row)
            (launch.block*p.block+lane.val) (x (launch.block*p.block+lane.val))
            (KernelTensor.logicalReal interpret words views.lse launch.row) (KernelTensor.logicalReal interpret words views.upstream launch.row))) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠launch.row*p.logitsStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  constructor
  · exact KernelBoundary.backward_addresses p views alloc bytes words inputs launch
  · exact KernelTensor.backward interpret p views alloc bytes words inputs launch pids

#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
end Real
namespace Native
open KernelProfile

def ScalarObservations (p : Parameters) (o : KernelABI.Options p) (C : KernelABI.Conversions)
    (I : Compiler.Instructions) (S : Compiler.Schedule) : Prop :=
  (p.hasSoftcap=Bool.true → Compiler.observe I S (.literal (some .fp32) .real p.softcap)=KernelABI.convert C o.softcap) ∧
  (p.hasScaling=Bool.true → Compiler.observe I S (.literal (some .fp32) .real p.scaling)=KernelABI.convert C o.scaling)

 theorem forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p 1) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (warps : Nat) (geometry : KernelLaunch.ForwardGeometry p warps)
    (C : KernelABI.Conversions) (abi : KernelABI.LiteralProfile p inputs.options C I)
    (launch : Launch p 1) (pids : LaunchState launch words) :
    ScalarObservations p inputs.options C I S ∧
    KernelBoundary.ForwardAddresses p views alloc bytes words inputs launch ∧
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.forward p) initial=some final ∧
      Compiler.observe I S (tensorValue final views.lse launch.row)=
        Compiler.observe I S (forwardLSE R p words 0) ∧
      Compiler.observe I S (tensorValue final views.loss launch.row)=
        Compiler.observe I S (fullLoss R p words) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row) → final.mem region offset=initial.mem region offset) := by
  refine ⟨⟨KernelProfile.literal_softcap p inputs.options C I S abi,
    KernelProfile.literal_scaling p inputs.options C I S abi⟩,?_,?_⟩
  · exact KernelBoundary.forward_addresses p views alloc bytes words inputs launch
  · exact KernelProfile.forward R I S p views alloc bytes words inputs launch pids

 theorem chunked (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p p.chunks) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (warps : Nat) (geometry : KernelLaunch.ChunkedGeometry p warps)
    (C : KernelABI.Conversions) (abi : KernelABI.LiteralProfile p inputs.options C I)
    (launch : Launch p p.chunks) (pids : LaunchState launch words) :
    ScalarObservations p inputs.options C I S ∧
    KernelBoundary.ForwardAddresses p views alloc bytes words inputs launch ∧
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.chunked p) initial=some final ∧
      Compiler.observe I S (tensorValue final views.lse (launch.row*p.chunks+launch.block))=
        Compiler.observe I S (forwardLSE R p words launch.block) ∧
      (launch.block=0 → Compiler.observe I S (tensorValue final views.loss launch.row)=
        Compiler.observe I S (partialLoss R p words)) ∧
      (∀ region offset, (launch.block=0 → region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row*p.chunks+launch.block) →
        final.mem region offset=initial.mem region offset) := by
  refine ⟨⟨KernelProfile.literal_softcap p inputs.options C I S abi,
    KernelProfile.literal_scaling p inputs.options C I S abi⟩,?_,?_⟩
  · exact KernelBoundary.forward_addresses p views alloc bytes words inputs launch
  · exact KernelProfile.chunked R I S p views alloc bytes words inputs launch pids

 theorem backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (warps : Nat) (geometry : KernelLaunch.BackwardGeometry p blocks warps)
    (C : KernelABI.Conversions) (abi : KernelABI.LiteralProfile p inputs.options C I)
    (launch : Launch p blocks) (pids : LaunchState launch words) :
    ScalarObservations p inputs.options C I S ∧
    KernelBoundary.BackwardAddresses p views alloc bytes words inputs launch ∧
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
        Compiler.observe I S (tensorValue final views.logits
          (launch.row*p.vocab+(launch.block*p.block+lane.val)))=
        Compiler.observe I S (backwardLane R p words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠launch.row*p.logitsStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  refine ⟨⟨KernelProfile.literal_softcap p inputs.options C I S abi,
    KernelProfile.literal_scaling p inputs.options C I S abi⟩,?_,?_⟩
  · exact KernelBoundary.backward_addresses p views alloc bytes words inputs launch
  · exact KernelProfile.backward R I S p views alloc bytes words inputs launch pids

#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
end Native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelPublic
