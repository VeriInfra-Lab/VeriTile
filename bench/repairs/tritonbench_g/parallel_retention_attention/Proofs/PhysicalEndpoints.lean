import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelRealTarget
import bench.repairs.tritonbench_g.parallel_retention_attention.Kernels
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelWordTarget
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicPhysicalTransport

/- Exact previous physical endpoints. PublicPhysicalTransport proves that their
source execution is the shared public IO schedule followed by these same masked
physical commits; arbitrary input aliases and exact exterior frames remain. -/

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Physical.Real
open VeriTile.Triton
open Kernel Kernel.Native
noncomputable section

/-- The shared DSL terminates, stores the independent causal shard formula
at every active physical word, and preserves every exterior word. -/
theorem forward (abi : LocalRuntime ℝ σ) (p : DotTrees) (s : Shape) (b : Bases)
    (tile : Tile) (output : Nat) (contract : ForwardContract s b tile output) (state : σ) :
    ∃ final,executeForward abi realScalar p s b tile output state=some final ∧
      (∀ i : TileIndex [128,s.BV],activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i →
        abi.observe final (output+forwardAddress s tile i)=
          RealTarget.forward s (currentInputs realOps s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ address,address∉forwardFootprint s tile output → abi.observe final address=abi.observe state address) := by
  obtain ⟨final,run,read,frame⟩ := forward_words abi realScalar p s b tile contract.program output state
  refine ⟨final,run,?_,frame⟩
  intro i hi
  rw [read i hi,realScalar_ops]
  exact forward_causal (plans s p) s _ tile i.1.val i.2.1.val i.1.isLt

/-- One complete backward source execution stores all three differentiated
causal shard formulas. No execution or output is a premise. -/
theorem backward (abi : LocalRuntime ℝ σ) (p : DotTrees) (s : Shape) (b : Bases)
    (tile : Tile) (dq dk dv : Nat) (contract : BackwardContract s b tile dq dk dv) (state : σ) :
    ∃ final,executeBackward abi realScalar p s b tile dq dk dv state=some final ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        abi.observe final (dq+keyAddress s tile i)=
          RealTarget.query s (currentInputs realOps s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        abi.observe final (dk+keyAddress s tile i)=
          RealTarget.key s (currentInputs realOps s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ i : TileIndex [64,s.BV],activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i →
        abi.observe final (dv+valueAddress s tile i)=
          RealTarget.value s (currentInputs realOps s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ address,address∉keyFootprint s tile dq ∪ keyFootprint s tile dk ∪ valueFootprint s tile dv →
        abi.observe final address=abi.observe state address) := by
  obtain ⟨final,run,hq,hk,hv,frame⟩ := backward_words abi realScalar p s b tile contract.program
    contract.shape.indices.bk_pos contract.shape.indices.bv_pos dq dk dv contract.separate state
  refine ⟨final,run,?_,?_,?_,frame⟩
  · intro i hi
    rw [hq i hi,realScalar_ops]
    exact query_causal (plans s p) s _ tile i.1.val i.2.1.val i.1.isLt
  · intro i hi
    rw [hk i hi,realScalar_ops,keyValue_causal (plans s p) s _ tile i.1.val i.2.1.val 0 i.1.isLt]
    rfl
  · intro i hi
    rw [hv i hi,realScalar_ops,keyValue_causal (plans s p) s _ tile i.1.val 0 i.2.1.val i.1.isLt]
    rfl

#axiomsClean forward
#axiomsClean backward
#specNonCircular RealTarget.forward avoiding [executeForward,forwardLane]
#specNonCircular RealTarget.query avoiding [executeBackward,queryLane]
#specNonCircular RealTarget.key avoiding [executeBackward,keyValueLane]
#specNonCircular RealTarget.value avoiding [executeBackward,keyValueLane]
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Physical.Real

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Physical.Native
open VeriTile.Triton
open Kernel Kernel.Native NativeBytes
noncomputable section

/-! Complete native kernel interfaces over raw entry bytes and word addresses.
Scalar FP32 operations and finite dot instruction trees retain their existing
primitive/compiler meanings. Tensor admission and readback are in TensorBridge. -/
theorem forward (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (tile : Tile) (output : Nat) (contract : ForwardContract s b tile output) (entry : Bytes) :
    ∃ final, executeForward NativeBytes.runtime native p s b tile output entry = some final ∧
      (∀ i : TileIndex [128,s.BV], activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i →
        NativeBytes.load final (output+forwardAddress s tile i) =
          WordTarget.forward native p s b entry tile i.1.val i.2.1.val) ∧
      (∀ address, address ∉ forwardFootprint s tile output →
        NativeBytes.load final address = NativeBytes.load entry address) := by
  exact forward_words NativeBytes.runtime native p s b tile contract.program output entry

theorem backward (native : Scalar FP32Word) (p : DotTrees) (s : Shape) (b : Bases)
    (tile : Tile) (dq dk dv : Nat) (contract : BackwardContract s b tile dq dk dv) (entry : Bytes) :
    ∃ final, executeBackward NativeBytes.runtime native p s b tile dq dk dv entry = some final ∧
      (∀ i : TileIndex [64,s.BK], activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        NativeBytes.load final (dq+keyAddress s tile i) =
          WordTarget.query native p s b entry tile i.1.val i.2.1.val) ∧
      (∀ i : TileIndex [64,s.BK], activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        NativeBytes.load final (dk+keyAddress s tile i) =
          WordTarget.key native p s b entry tile i.1.val i.2.1.val) ∧
      (∀ i : TileIndex [64,s.BV], activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i →
        NativeBytes.load final (dv+valueAddress s tile i) =
          WordTarget.value native p s b entry tile i.1.val i.2.1.val) ∧
      (∀ address, address ∉ keyFootprint s tile dq ∪ keyFootprint s tile dk ∪ valueFootprint s tile dv →
        NativeBytes.load final address = NativeBytes.load entry address) := by
  exact backward_words NativeBytes.runtime native p s b tile contract.program
    contract.shape.indices.bk_pos contract.shape.indices.bv_pos dq dk dv contract.separate entry

#axiomsClean forward
#axiomsClean backward
#specNonCircular WordTarget.forward avoiding [executeForward,FP.Structural.exec]
#specNonCircular WordTarget.query avoiding [executeBackward,FP.Structural.exec]
#specNonCircular WordTarget.key avoiding [executeBackward,FP.Structural.exec]
#specNonCircular WordTarget.value avoiding [executeBackward,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Physical.Native
