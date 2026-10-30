import bench.tritonbench_g.llama_ff_triton.Proofs.Iteration

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem bodyInvariant {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant a native p initial k current) (bytes : ByteValues initial) :
    Invariant a native p initial (k+1) (bodyExecution a native p initial current k h bytes).val := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  have mask (lane : Nat) : ((lane : Int)<(K : Int)-(k : Int)*64) ↔ k*64+lane<K := by omega
  cases native <;> cases fp8
  all_goals constructor
  all_goals try constructor
  all_goals simp [bodyExecution,h.memory,h.m,h.n,h.k,h.sak,h.w1k,h.w3k,h.rms,
    h.outm,h.outn,h.pidm,h.pidn,h.offk,h.aptrs,h.bptrs,h.cptrs,h.rptrs,
    h.acc1,h.acc2,h.sum,aOffset,wOffset,accumulation,scaledValue,inputValue,rmsValue,
    weightValue,nativeWeightValue,rmsByte,weightByte,halfZero,activeK,mask,moment,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_mul,Nat.add_assoc]
  all_goals funext i
  all_goals rcases i with ⟨i,j,⟨⟩⟩
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    decide_eq_true_eq,mask,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try delta scaledValue inputValue rmsValue weightValue nativeWeightValue rmsByte weightByte
  all_goals try simp [activeK,halfZero,aOffset,wOffset,mask,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals norm_num
  all_goals rfl

#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
