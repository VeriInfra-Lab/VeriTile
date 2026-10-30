import bench.tritonbench_g.chunk_delta_fwd.Sources.Inner
import bench.tritonbench_g.chunk_delta_fwd.Sources.Range
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem innerLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p (fun _ => zero M) current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c)) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i_c" 0 (innerCount p) 1
      (inner p).body current=some t ∧ Ready pid input t ∧ HasH p frozen t ∧
      HasAcc p (accumulated M D p pid input c frozen) t ∧
      t.mem=writeInner M D p pid input c frozen current.mem ∧
      t.regs .nat [] "i_t"=some (fun _ => c) := by
  let ap := fun n => foldRange (fun b acc => accumulatedStep M D p pid input c b frozen acc)
    0 n 1 (fun _ => zero M)
  let mp := fun n => foldRange (fun b mem => writeCorrected M p pid c b
    (corrected M D p pid input c b frozen) mem) 0 n 1 current.mem
  let I := fun n (s : State α) => Ready pid input s ∧ HasH p frozen s ∧
    HasAcc p (ap n) s ∧ s.mem=mp n ∧ s.regs .nat [] "i_t"=some (fun _ => c)
  have hs : ∀ n, n < innerCount p → ∀ s, I n s →
      ∃ t, run (engine (withDot M D)) (inner p).body
        (s.setReg "i_c" .nat [] (fun _ => n))=some t ∧ I (n+1) t := by
    intro n _ s hi
    obtain ⟨ctx,hh,ha,hm,ht⟩ := hi
    let e := innerExecution M D p pid input s c n frozen (ap n) ctx hh ha ht
    have hv := inner_values M D p pid input s c n frozen (ap n) ctx hh ha ht
    refine ⟨e.val,e.property,hv.1,hv.2.1,?_,?_,hv.2.2.2⟩
    · simpa only [ap,foldRange_succ] using hv.2.2.1
    · dsimp only [e]
      rw [inner_memory,hm]
      exact (foldRange_succ _ n _).symm
  have hi : I 0 current := by
    exact ⟨ctx,hh,by simpa only [ap,foldRange_zero] using ha,
      by simp only [mp,foldRange_zero],ht⟩
  obtain ⟨t,he,hi⟩ := range_invariant (engine (withDot M D)) "i_c" (inner p).body
    (innerCount p) I hs 0 (Nat.zero_le _) current hi
  exact ⟨t,he,hi.1,hi.2.1,hi.2.2.1,hi.2.2.2.1,hi.2.2.2.2⟩

#axiomsClean innerLoop
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
