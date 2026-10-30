import bench.tritonbench_g.llama_ff_triton.Proofs.IterationInvariant
import bench.tritonbench_g.llama_ff_triton.Proofs.Output

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem loopExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State α) (k : Nat) (hk : k≤steps p)
    (h : Invariant a native p initial k current) (bytes : ByteValues initial) :
    ∃ final, rangeInt a.model "step" k (steps p) 1 (loopBody native p) current=some final ∧
      Invariant a native p initial (steps p) final := by
  have aux : ∀ n k (current : State α), steps p-k=n → k≤steps p →
      Invariant a native p initial k current →
      ∃ final, rangeInt a.model "step" k (steps p) 1 (loopBody native p) current=some final ∧
        Invariant a native p initial (steps p) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k<steps p
      · let next := bodyExecution a native p initial current k h bytes
        have hi := bodyInvariant a native p initial current k h bytes
        obtain ⟨final,he,hf⟩ := ih (steps p-(k+1)) (by omega)
          (k+1) next.val rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [rangeInt,if_neg (by decide),if_pos (by exact_mod_cast hlt),next.property]
        simpa only [Option.bind_some,Int.natCast_add,Int.natCast_one] using he
      · have heq : k=steps p := by omega
        subst k
        refine ⟨current,?_,h⟩
        rw [rangeInt,if_neg (by decide),if_neg (by omega)]
  exact aux _ _ _ rfl hk h

theorem countEvaluation {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (current : State α)
    (hk : current.regs .int [] "K64"=some (fun _ => (p.inner : Int))) :
    evalOp a.model none countOp current=some (fun _ => (steps p : Int)) := by
  have hn : ¬(p.inner : Int)+64<1 := by omega
  have hc : ((p.inner : Int)+64-1)/64=(steps p : Int) := by
    have hd := signed_cdiv p.inner 64 (by decide)
    rw [Int.tdiv_eq_ediv_of_nonneg (by omega)] at hd
    simpa [steps,Nat.add_sub_assoc] using hd
  simp [countOp,evalOp_unfold,hk,numeric,numericLt,bop,integralFloorDiv,hn,hc]

#axiomsClean loopExecution
#axiomsClean countEvaluation
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
