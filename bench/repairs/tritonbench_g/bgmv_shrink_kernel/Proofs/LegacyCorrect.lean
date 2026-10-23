/- Historical padded/snapshot wrapper theorem, retained proof credit. -/
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealHost
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealWorkset
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealFlat
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.NativeWords
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.HostProtocol

/-! Reader entry for the separate repaired candidate. The independent target is
caller Y_entry[b,n]+scaling*sum_{k<K} X_entry[b,k]*W_entry[index[b],n,k].
The -1 sentinel preserves its row. Source arithmetic is interpreted over Real;
FP rounding and actual atomic order are handled independently in FPEquiv.
The production profile preserves both original FP16/BF16 input branches and
uses FP32 output by the user's default for an unspecified role. The generic
formal output parameter is not a claim about integer or FP64 outputs.

The wrapper admits arbitrary logical read views and role aliases; nonoverlap
of the supplied output follows from its exact gcd guard. B≤input/output rows,
N≤output columns, integer metadata, valid selected adapters and common device
are explicit wrapper checks. Mathematical dimensions have no small-shape cap.
Native bounds come from successful padded allocations; generic launch/PID
compatibility remains the global runtime obligation. The actual launch is one
(64,B) grid. Orders are arbitrary separately per cell, never serialized launches.
-/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Correct
open VeriTile.Triton Snapshot CallerOutput
noncomputable section

abbrev mathematicalTarget := RealHost.math
abbrev sourcePrograms := RealHost.every_program
abbrev flatPrograms := RealFlat.complete
abbrev exactEventCoverage := AtomicOrder.fiber
abbrev generatedNativeAddresses := NativeWords.linked
abbrev sourceAtomicEvent := RealWorkset.source_event
abbrev protocolFrame := @HostProtocol.frame
abbrev protocolIdentity := @HostProtocol.identity

def scratch (h : Call) (orders : AtomicOrder.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (b n : Nat) : ℝ :=
  if inside : b<h.batches ∧ n<h.N then
    RealHost.scratchOutput h orders logical packed ⟨b,inside.1⟩ ⟨n,inside.2⟩
  else logical.output b n

/-- Successful logical index_copy_ returns None, preserves the exact supplied
view/object, writes the independent Real target, and frames all other bytes.
The copied scratch values are the proved source payloads composed in the actual
atomic linearization, not a numerical output assumption. -/
theorem returned_correct (h : Call) (orders : AtomicOrder.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0≤logical.indices b ∧ (logical.indices b).toNat<h.adapters))
    (v : View) (guard : Shared.Admitted v.original) (encode : Bool → ℝ → Fin v.width → BitVec 8)
    (before : Bytes) :
    let result := returned h v encode logical.indices (scratch h orders logical packed) before
    result.result=() ∧ result.view=v ∧
    (∀ i : Coord v,Active h v logical.indices i → ∀ byte,
      result.memory v.original.owner (address v i byte)=
        encode v.original.negative (mathematicalTarget h logical i.1.val i.2.val) byte) ∧
    (∀ owner ptr,(¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v logical.indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) →
      result.memory owner ptr=before owner ptr) := by
  refine ⟨rfl,rfl,?_,?_⟩
  · intro i active byte
    rw [show (returned h v encode logical.indices (scratch h orders logical packed) before).memory=
      copy h v encode logical.indices (scratch h orders logical packed) before from rfl]
    rw [copy_byte h v encode guard logical.indices _ before i active byte]
    unfold scratch
    rw [dif_pos ⟨active.1,active.2.1⟩,RealHost.readback h orders logical packed valid]
  · exact CallerOutput.frame h v encode logical.indices _ before

#axiomsClean returned_correct

/-- Read the values produced by the local atomic traces from their actual
fresh output-allocation addresses, before the logical tensor copy. -/
def runtimeScratch (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (run : RealWorkset.Run h orders logical packed)
    (storage : Storage.Placement h) (heap : Nat → ℝ) (b n : Nat) : ℝ :=
  if inside : b<h.batches ∧ n<h.N then
    RealWorkset.physical h orders logical packed run storage heap
      (Physical.outputAddress storage (⟨b,inside.1⟩,⟨n,inside.2⟩))
  else logical.output b n

/-- Whole-workset returned-byte composition. Source event execution is proved
by `sourceAtomicEvent`; runtime traces carry only local atomic transitions.
Physical readback is derived, then the original supplied view is updated. -/
theorem returned_from_runtime (h : Call) (orders : Workset.Orders h) (logical : Logical ℝ)
    (packed : Packed ℝ h logical 0) (run : RealWorkset.Run h orders logical packed)
    (valid : ∀ b,b<h.batches → logical.indices b=-1 ∨
      (0≤logical.indices b ∧ (logical.indices b).toNat<h.adapters))
    (storage : Storage.Placement h) (heap : Nat → ℝ)
    (v : View) (guard : Shared.Admitted v.original) (encode : Bool → ℝ → Fin v.width → BitVec 8)
    (before : Bytes) :
    RealWorkset.SourceWorkset h logical packed ∧
    (let result := returned h v encode logical.indices
      (runtimeScratch h orders logical packed run storage heap) before
    result.result=() ∧ result.view=v ∧
    (∀ i : Coord v,Active h v logical.indices i → ∀ byte,
      result.memory v.original.owner (address v i byte)=
        encode v.original.negative (mathematicalTarget h logical i.1.val i.2.val) byte) ∧
    (∀ owner ptr,(¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v logical.indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) →
      result.memory owner ptr=before owner ptr)) := by
  constructor
  · exact RealWorkset.source_workset h logical packed (fun b hb =>
      (valid b hb).imp_right And.left)
  refine ⟨rfl,rfl,?_,?_⟩
  · intro i active byte
    change copy h v encode logical.indices _ before v.original.owner (address v i byte)=_
    rw [copy_byte h v encode guard logical.indices _ before i active byte]
    unfold runtimeScratch
    rw [dif_pos ⟨active.1,active.2.1⟩]
    rw [RealWorkset.physical_readback h orders logical packed run valid storage heap _ _ active.2.2]
  · exact CallerOutput.frame h v encode logical.indices _ before

#axiomsClean returned_from_runtime
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Correct
