import VeriTile.Triton.Float.StructuralLoopNormalization
import bench.repairs.tritonbench_g.chunk_delta_fwd.OuterEdges
import bench.repairs.tritonbench_g.chunk_delta_fwd.InnerLoop
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem outerStep (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current) :
    ∃ t, run (engine (withDot M D)) (outer p).body
      (current.setReg "i_t" .nat [] (fun _ => c))=some t ∧
      Ready pid input t ∧ HasH p (next M D p pid input c frozen) t ∧
      t.mem=writeInner M D p pid input c frozen (writeHistory M p pid c frozen current.mem) := by
  let start := historyExecution M D p pid input current c frozen ctx hh
  have hv := history_values M D p pid input current c frozen ctx hh
  obtain ⟨mid,hm,hctx,hhmid,hamid,hmem,_⟩ := innerLoop M D p pid input start.val c frozen
    hv.1 hv.2.1 hv.2.2.1 hv.2.2.2
  let finish := updateState M p mid frozen (accumulated M D p pid input c frozen)
  have hf := update_run M D p mid frozen (accumulated M D p pid input c frozen) hhmid hamid
  refine ⟨finish,?_,update_ready M p pid input mid _ _ hctx,?_,?_⟩
  · have hl : step (engine (withDot M D))
        (.forLoop "i_c" ((p.BT + p.BC - 1) / p.BC) (inner p).body) start.val=some mid := by
      simpa only [step, loop_eq_range, innerCount] using hm
    rw [inner_parts,List.append_assoc,run_append,start.property]
    simp only [Option.bind_some,List.singleton_append,run,hl,bind,hf]
    rfl
  · simp [HasH,finish,updateState]
    rfl
  · change mid.mem=_
    rw [hmem]
    dsimp only [start]
    rw [history_memory]

theorem outerLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α)
    (ctx : Ready pid input current) (hh : HasH p (initial M p pid input) current) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i_t" 0 p.NT 1 (outer p).body current=some t ∧
      Ready pid input t ∧ HasH p (states M D p pid input p.NT) t ∧
      t.mem=foldRange (writeOuter M D p pid input) 0 p.NT 1 current.mem := by
  let mp := fun n => foldRange (writeOuter M D p pid input) 0 n 1 current.mem
  let I := fun n (s : State α) => Ready pid input s ∧
    HasH p (states M D p pid input n) s ∧ s.mem=mp n
  have hs : ∀ n, n < p.NT → ∀ s, I n s →
      ∃ t, run (engine (withDot M D)) (outer p).body
        (s.setReg "i_t" .nat [] (fun _ => n))=some t ∧ I (n+1) t := by
    intro n _ s hi
    obtain ⟨t,he,hc,hh,hm⟩ := outerStep M D p pid input s n
      (states M D p pid input n) hi.1 hi.2.1
    refine ⟨t,he,hc,hh,?_⟩
    rw [hm,hi.2.2]
    exact (foldRange_succ (writeOuter M D p pid input) n current.mem).symm
  have hi : I 0 current := ⟨ctx,hh,by simp only [mp,foldRange_zero]⟩
  obtain ⟨t,he,hi⟩ := range_invariant (engine (withDot M D)) "i_t" (outer p).body
    p.NT I hs 0 (Nat.zero_le _) current hi
  exact ⟨t,he,hi⟩

#axiomsClean outerStep
#axiomsClean outerLoop
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
