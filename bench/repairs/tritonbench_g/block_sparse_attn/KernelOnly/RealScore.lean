import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.RealArithmetic
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.ChannelLoop

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def UniformMatrix (x : Matrix α) : Prop :=
  ∀ i o, x i o=x i (owner i.2.1.val)

theorem scoreStep_uniform {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (column : Int) (old : Matrix α)
    (uniform : UniformMatrix old) (block : Nat) :
    UniformMatrix (scoreStep a native p s column old block) := by
  intro i o
  cases native <;> simp [scoreStep,tensor,broadcast,binary,uniform i o]

theorem scoreAccumulation_uniform {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (column : Int) (blocks : Nat) :
    UniformMatrix (scoreAccumulation a native p s column blocks) := by
  induction blocks with
  | zero => intro i o; rfl
  | succ blocks ih =>
    rw [scoreAccumulation_succ]
    exact scoreStep_uniform a native p s column _ ih blocks

theorem scoreStep_math (p : Parameters) (s : State (Replicas R)) (column : Int)
    (old : Matrix R) (uniform : UniformMatrix old) (block : Nat) :
    scoreStep mathArithmetic Bool.false p s column old block=
      scoreStep mathArithmetic Bool.true p s column old block := by
  funext i o
  simp only [scoreStep,Bool.false_eq_true,ite_false,ite_true,tensor,broadcast,binary]
  rw [uniform i o]
  exact (math_chain_add _ _ _).symm

/-- This equality is proved only in the independent exact-Real interpretation;
the native theorem retains the selected seeded MMA graph instead. -/
theorem scoreAccumulation_math (p : Parameters) (s : State (Replicas R)) (column : Int)
    (blocks : Nat) : scoreAccumulation mathArithmetic Bool.false p s column blocks=
      scoreAccumulation mathArithmetic Bool.true p s column blocks := by
  induction blocks with
  | zero => rfl
  | succ blocks ih =>
    rw [scoreAccumulation_succ,scoreAccumulation_succ,ih]
    exact scoreStep_math p s column _ (scoreAccumulation_uniform mathArithmetic Bool.true p s column blocks) blocks

#axiomsClean scoreStep_uniform
#axiomsClean scoreAccumulation_uniform
#axiomsClean scoreStep_math
#axiomsClean scoreAccumulation_math
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
