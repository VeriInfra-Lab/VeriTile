import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealSetupSource

/-! Execution of the actual signed-metadata prefix. A nonnegative slot starts
the split-K loop with zero accumulator and unchanged memory. The sentinel is
handled separately; no natural-valued substitute metadata is introduced. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetup
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def afterPrelude (s : BlockState) (slot : Int) : BlockState :=
  ((s.setReg "pid_sk" .nat [] (Tile.scalar (s.pids 0))).setReg
    "cur_batch" .nat [] (Tile.scalar (s.pids 1))).setReg
    "lora_index" .int [] (Tile.scalar slot)

theorem guard_eval (s : BlockState) (slot : Int)
    (h : s.regs .int [] "lora_index" = some (Tile.scalar slot)) :
    evalOp Parameters.guard s = some (Tile.scalar (decide (slot ≠ -1))) := by
  simp only [Parameters.guard, evalOp, h, Tile.cop, Tile.scalar_data,
    Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

private def offsets (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  ((afterPrelude initial (Int.ofNat slot)).setReg "offset_n" .nat [p.BN]
    (Tile.vec (fun n => n.val))).setReg "offset_k" .nat [p.BK]
    (Tile.vec (fun k => k.val + initial.pids 0 * p.BK))

private def inputState (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (offsets p initial slot).setReg "a_ptr" .ptr []
    (Tile.scalar (p.input, initial.pids 1 * p.inputBatchStride))

private def weightState (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (inputState p initial slot).setReg "b_ptr" .ptr []
    (Tile.scalar (p.weights, p.slotStride * slot))

def afterSetup (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (weightState p initial slot).setReg "accumulator" .real [p.BN] ⟨fun _ => some 0⟩

private theorem weight_eval (p : Parameters) (s : BlockState) (slot : Nat)
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat slot))) :
    evalOp p.weightPointer s = some (Tile.scalar (p.weights, p.slotStride * slot)) := by
  simp only [Parameters.weightPointer, evalOp, hlx, Tile.bop, Tile.uop,
    Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro i
  change (p.weights, 0 + (Int.ofNat p.slotStride * Int.ofNat slot).toNat) =
    (p.weights, p.slotStride * slot)
  simp only [Nat.zero_add, Int.ofNat_eq_natCast, ← Nat.cast_mul, Int.toNat_natCast]

@[simp] theorem setup_mem (p : Parameters) (initial : BlockState) (slot : Nat) :
    (afterSetup p initial slot).mem = initial.mem := rfl

@[simp] theorem setup_pids (p : Parameters) (initial : BlockState) (slot : Nat) :
    (afterSetup p initial slot).pids = initial.pids := rfl

/-- The actual signed source prefix establishes the invariant for arbitrary
nonnegative slots, program IDs and dimensions, including a masked K tail. -/
theorem setup_invariant (p : Parameters) (initial : BlockState) (slot : Nat)
    (hS : 0 < p.BK * p.splits) :
    RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
      p.rowStride p.colStride p.BN p.BK p.splits initial 0 (afterSetup p initial slot) := by
  simp [RealLoop.shrinkInv, afterSetup, weightState, inputState, offsets, afterPrelude,
    RealLoop.accPartial, Nat.add_pos_right p.K hS]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetup
