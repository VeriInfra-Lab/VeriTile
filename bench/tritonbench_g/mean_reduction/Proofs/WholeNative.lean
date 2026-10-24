import bench.tritonbench_g.mean_reduction.Proofs.WholeFPValue

namespace VeriTile.Bench.TritonBenchG.MeanReduction.WholeNative
open VeriTile.Triton FP.Structural
noncomputable section

/-- Logical column selected by a native butterfly shuffle. The modulo merely
makes this a total finite function; all recorded masks stay within eight lanes. -/
def xorLane (i : Fin 8) (mask : Nat) : Fin 8 := ⟨(i.val ^^^ mask)%8,by omega⟩

def butterfly (add : α → α → α) (values : Fin 8 → α) (mask : Nat) : Fin 8 → α :=
  fun i => add (values i) (values (xorLane i mask))

/-- Dynamic-N lowering: own-left butterfly masks4,2,1. Every physical replica
that writes the same shared cell is retained as a separate local tree. -/
def orderedDynamic (add : α → α → α) (values : Fin 8 → α) (lane : Fin 8) : α :=
  butterfly add (butterfly add (butterfly add values 4) 2) 1 lane

/-- N=1 specialization: within-warp masks16,8 correspond to logical masks2,1;
selected partials for columns0..3 and4..7 are combined left-to-right. -/
def orderedOne (add : α → α → α) (values : Fin 8 → α) : α :=
  add (add (add (values 0) (values 2)) (add (values 1) (values 3)))
    (add (add (values 4) (values 6)) (add (values 5) (values 7)))

theorem dynamic_zero (add : α → α → α) (values : Fin 8 → α) :
    orderedDynamic add values 0 =
      add (add (add (values 0) (values 4)) (add (values 2) (values 6)))
        (add (add (values 1) (values 5)) (add (values 3) (values 7))) := rfl

theorem xor_exact (i : Fin 8) (mask : Nat) (hm : mask∈[1,2,4]) :
    (xorLane i mask).val=i.val ^^^ mask := by
  fin_cases i <;> simp only [List.mem_cons,List.not_mem_nil,or_false] at hm
  all_goals rcases hm with rfl|rfl|rfl <;> decide

/-- Exact one-iteration source recurrence for the ordinary Python N=1
specialization. Its only addition has the source operands +0 then loaded value;
the native operand reversal is a separate local numerical binding. -/
theorem accumulated_one {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName)
    (M pid : Nat) (memory : RegionName → Nat → Cell α) (acc : WholeFPValue.Acc α) :
    WholeFPValue.accumulated A X M 1 pid memory 0 acc =
      WholeFPValue.nextAcc A X M 1 pid 0 memory acc := by
  rw [WholeFPValue.accumulated]
  simp only [show (0:Nat)<1 by omega,dif_pos]
  rw [WholeFPValue.accumulated]
  simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.WholeNative
