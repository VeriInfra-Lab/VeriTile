import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PhysicalEndpoints
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelWordTarget
import bench.repairs.tritonbench_g.parallel_retention_attention.Kernels
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelByteReadback

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.TensorBridge
open VeriTile.Triton
open Kernel Kernel.Native NativeBytes
noncomputable section

/-! FP32 source-precision closure. Native scalar operations are elementary
primitive models, including the selected log2/exp2 behavior and FP32 FMA.
DotTrees carries finite instruction syntax and structural coverage only.
Local dot→dotAcc staging preserves the live carry and FMA order. These are
primitive/compiler model boundaries, not assumptions about whole execution,
output agreement, or numerical experiments. There is no finite-input guard. -/

theorem forward (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (observed : Checked.Inputs) (admitted : Checked.admit s observed=Bool.true)
    (prepared : Checked.Prepared s observed b) (tile : Tile) (output : Nat)
    (contract : ForwardContract s b tile output)
    (tensor : ScratchTensor s .forward output observed.query.metadata.device.index) (entry : Bytes) :
    ∃ final,executeForward NativeBytes.runtime native p s b tile output entry=some final ∧
      (∀ i : TileIndex [128,s.BV],activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i →
        ∀ j : Fin 4,
        final (tensor.metadata.dataPtr.toNat+4*forwardAddress s tile i+j.val)=
          encode (WordTarget.forward native p s b entry tile i.1.val i.2.1.val) j ∧
        tensor.metadata.dataPtr.toNat+4*forwardAddress s tile i+j.val<2^64) ∧
      (∀ byte,byte/4∉forwardFootprint s tile output → final byte=entry byte) := by
  obtain ⟨final,run,read,frame⟩ := Physical.Native.forward native p s b tile output contract entry
  refine ⟨final,run,?_,?_⟩
  · intro i hi j
    exact ⟨scratch_byte s .forward output _ tensor final _ _ (read i hi) j,
      scratch_byte_bound s .forward output _ tensor _ (forward_address_bound s tile contract.program i hi) j⟩
  · intro byte outside
    exact frame_of_words entry final byte (frame _ outside)

theorem backward (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (observed : Checked.Inputs) (admitted : Checked.admit s observed=Bool.true)
    (prepared : Checked.Prepared s observed b)
    (gradient : Cotangent s b observed.query.metadata.device.index)
    (tile : Tile) (dq dk dv : Nat) (contract : BackwardContract s b tile dq dk dv)
    (tq : ScratchTensor s .query dq observed.query.metadata.device.index)
    (tk : ScratchTensor s .key dk observed.query.metadata.device.index)
    (tv : ScratchTensor s .value dv observed.query.metadata.device.index) (entry : Bytes) :
    ∃ final,executeBackward NativeBytes.runtime native p s b tile dq dk dv entry=some final ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        ∀ j : Fin 4,
        final (tq.metadata.dataPtr.toNat+4*keyAddress s tile i+j.val)=
          encode (WordTarget.query native p s b entry tile i.1.val i.2.1.val) j ∧
        tq.metadata.dataPtr.toNat+4*keyAddress s tile i+j.val<2^64) ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        ∀ j : Fin 4,
        final (tk.metadata.dataPtr.toNat+4*keyAddress s tile i+j.val)=
          encode (WordTarget.key native p s b entry tile i.1.val i.2.1.val) j ∧
        tk.metadata.dataPtr.toNat+4*keyAddress s tile i+j.val<2^64) ∧
      (∀ i : TileIndex [64,s.BV],activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i →
        ∀ j : Fin 4,
        final (tv.metadata.dataPtr.toNat+4*valueAddress s tile i+j.val)=
          encode (WordTarget.value native p s b entry tile i.1.val i.2.1.val) j ∧
        tv.metadata.dataPtr.toNat+4*valueAddress s tile i+j.val<2^64) ∧
      (∀ byte,byte/4∉keyFootprint s tile dq ∪ keyFootprint s tile dk ∪ valueFootprint s tile dv →
        final byte=entry byte) := by
  obtain ⟨final,run,hq,hk,hv,frame⟩ := Physical.Native.backward native p s b tile dq dk dv contract entry
  refine ⟨final,run,?_,?_,?_,?_⟩
  · intro i hi j
    exact ⟨scratch_byte s .query dq _ tq final _ _ (hq i hi) j,
      scratch_byte_bound s .query dq _ tq _ (key_address_bound s tile contract.program i hi) j⟩
  · intro i hi j
    exact ⟨scratch_byte s .key dk _ tk final _ _ (hk i hi) j,
      scratch_byte_bound s .key dk _ tk _ (key_address_bound s tile contract.program i hi) j⟩
  · intro i hi j
    exact ⟨scratch_byte s .value dv _ tv final _ _ (hv i hi) j,
      scratch_byte_bound s .value dv _ tv _ (value_address_bound s tile contract.program i hi) j⟩
  · intro byte outside
    exact frame_of_words entry final byte (frame _ outside)

#axiomsClean forward
#axiomsClean backward
#specNonCircular WordTarget.forward avoiding [executeForward,FP.Structural.exec]
#specNonCircular WordTarget.query avoiding [executeBackward,FP.Structural.exec]
#specNonCircular WordTarget.key avoiding [executeBackward,FP.Structural.exec]
#specNonCircular WordTarget.value avoiding [executeBackward,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.TensorBridge
