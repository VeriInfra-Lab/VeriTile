import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.PublicData
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.LogicalCongruence

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

theorem inputs_agree (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (xs : (io mode a s p incoming finalGradient seed final).Inputs)
    (initial : State (Staged.Node α))
    (hin : (io mode a s p incoming finalGradient seed final).InputBinding xs initial) :
    Logical.Agreement mode s seed final
      (Quoted.inputs (KernelInterface.layout s incoming finalGradient) initial) (logicalInputs a s xs) := by
  constructor
  · intro h hh t ht k hk
    have hi := hin 0 (⟨h,hh⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) (by simp [inputActive])
    change (initial.mem "q" (rowAddress s s.DK h t k)).read .real=xs 0 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      contiguous4,Nat.max_eq_left (by omega : 1 ≤ s.T),Nat.max_eq_left (by omega : 1 ≤ s.DK),readWord,logicalInputs,array,hh,ht,hk,rowAddress,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hi
  · intro h hh t ht k hk
    have hi := hin 1 (⟨h,hh⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) (by simp [inputActive])
    change (initial.mem "k" (rowAddress s s.DK h t k)).read .real=xs 1 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      contiguous4,Nat.max_eq_left (by omega : 1 ≤ s.T),Nat.max_eq_left (by omega : 1 ≤ s.DK),readWord,logicalInputs,array,hh,ht,hk,rowAddress,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hi
  · intro h hh t ht v hv
    have hi := hin 2 (⟨h,hh⟩,⟨t,ht⟩,⟨v,hv⟩,PUnit.unit) (by simp [inputActive])
    change (initial.mem "v" (rowAddress s s.DV h t v)).read .real=xs 2 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      contiguous4,Nat.max_eq_left (by omega : 1 ≤ s.T),Nat.max_eq_left (by omega : 1 ≤ s.DV),readWord,logicalInputs,array,hh,ht,hv,rowAddress,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hi
  · intro hm h hh t ht v hv
    have hi := hin 3 (⟨h,hh⟩,⟨t,ht⟩,⟨v,hv⟩,PUnit.unit) (by simp [inputActive,hm])
    change (initial.mem "do" ((h/s.H)*incoming 0+(h%s.H)*incoming 1+v*incoming 3+t*incoming 2)).read .real=xs 3 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      readWord,logicalInputs,array,hh,ht,hv] using hi
  · intro hs h hh k hk v hv
    have hi := hin 4 (⟨h,hh⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) (by simp [inputActive,hs])
    change (initial.mem "initial_state" (stateOffset s h k v)).read .real=xs 4 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      readWord,logicalInputs,array,hh,hk,hv,stateOffset,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hi
  · intro hm hf h hh k hk v hv
    have hi := hin 5 (⟨h,hh⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) (by simp [inputActive,hm,hf])
    change (initial.mem "d_final_state" ((h/s.H)*finalGradient 0+(h%s.H)*finalGradient 1+k*finalGradient 2+v*finalGradient 3)).read .real=xs 5 _ at hi
    have hi := congrArg Staged.Node.value hi
    simpa [Quoted.inputs,KernelInterface.layout,KernelInterface.buffers,KernelInterface.descriptor,layout,
      readWord,logicalInputs,array,hh,hk,hv] using hi

#axiomsClean inputs_agree
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
