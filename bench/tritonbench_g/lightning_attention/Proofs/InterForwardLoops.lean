import bench.tritonbench_g.lightning_attention.Proofs.InterQueryBody
import bench.tritonbench_g.lightning_attention.Proofs.InterDeltaBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
theorem foldRange_id {β : Type} (n : Nat) (x : β) : foldRange (fun _ a => a) 0 n 1 x=x := by
  induction n with
  | zero => exact foldRange_zero _ _
  | succ n ih => rw [foldRange_succ,ih]

theorem querySubFold (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) :
    foldRange (fun sub mem => querySubMemory M D p head ((block*p.NC+sub)*p.CB) block carry mem)
      0 p.NC 1 m=queryBlockMemory M D p head block carry m := by
  by_cases hb : block=0
  · simp [querySubMemory,queryBlockMemory,hb,foldRange_id]
  · simp only [querySubMemory,queryBlockMemory,hb,if_false]

theorem interQueryLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head (block*p.NC*p.CB) row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block)) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "j" 0 p.NC 1 (interFQueryParts p).body s=some t ∧
      InterFReady p head ((block+1)*p.NC*p.CB) row2 input carry t ∧
      t.mem=queryBlockMemory M D p head block carry s.mem := by
  let update := fun sub mem => querySubMemory M D p head ((block*p.NC+sub)*p.CB) block carry mem
  let I := fun n (t : State α) =>
    InterFReady p head ((block*p.NC+n)*p.CB) row2 input carry t ∧
    t.regs .nat [] "i"=some (fun _ => block) ∧ t.mem=foldRange update 0 n 1 s.mem
  have hs : ∀ n, n<p.NC → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interFQueryParts p).body
        (t.setReg "j" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t ht
    let run := interQueryBody M D p head ((block*p.NC+n)*p.CB) row2 block n input carry t ht.1 ht.2.1
    have next := interQueryBody_ready M D p head ((block*p.NC+n)*p.CB) row2 block n input carry t ht.1 ht.2.1
    refine ⟨run.val,run.property,?_,next.2,?_⟩
    · simpa only [run,Nat.add_mul,Nat.one_mul,Nat.add_assoc] using next.1
    · change (interQueryBody M D p head _ _ _ _ _ _ _ _ _).val.mem=_
      rw [interQueryBody_memory,ht.2.2]
      exact (foldRange_succ update n s.mem).symm
  have initial : I 0 s := by exact ⟨by simpa using ctx,hi,by simp [foldRange_zero]⟩
  obtain ⟨t,he,hf⟩ := range_invariant (engine (withDot M D)) "j" (interFQueryParts p).body p.NC I hs 0 (Nat.zero_le _) s initial
  refine ⟨t,he,?_,?_⟩
  · simpa [Nat.add_mul,Nat.one_mul] using hf.1
  · exact hf.2.2.trans (querySubFold M D p head block carry s.mem)

theorem interDeltaLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 block : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 (block*p.NC*p.CB) input carry s)
    (ha : InterFDelta p (fun _ => zero M) s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "j" 0 p.NC 1 (interFDeltaParts p).body s=some t ∧
      InterFReady p head row1 ((block+1)*p.NC*p.CB) input carry t ∧
      InterFDelta p (queryBlockDelta M D p head block input) t ∧ t.mem=s.mem := by
  let update := fun sub acc => querySubDelta M D p head ((block*p.NC+sub)*p.CB) input acc
  let I := fun n (t : State α) =>
    InterFReady p head row1 ((block*p.NC+n)*p.CB) input carry t ∧
    InterFDelta p (foldRange update 0 n 1 (fun _ => zero M)) t ∧ t.mem=s.mem
  have hs : ∀ n, n<p.NC → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interFDeltaParts p).body
        (t.setReg "j" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t ht
    let acc := foldRange update 0 n 1 (fun _ => zero M)
    let run := interDeltaBody M D p head row1 ((block*p.NC+n)*p.CB) n input carry acc t ht.1 ht.2.1
    have next := interDeltaBody_ready M D p head row1 ((block*p.NC+n)*p.CB) n input carry acc t ht.1 ht.2.1
    refine ⟨run.val,run.property,?_,?_,?_⟩
    · simpa only [run,Nat.add_mul,Nat.one_mul,Nat.add_assoc] using next.1
    · rw [foldRange_succ]
      exact next.2
    · exact (interDeltaBody_memory M D p head row1 _ n input carry acc t ht.1 ht.2.1).trans ht.2.2
  have initial : I 0 s := by exact ⟨by simpa using ctx,by simpa [foldRange_zero] using ha,rfl⟩
  obtain ⟨t,he,hf⟩ := range_invariant (engine (withDot M D)) "j" (interFDeltaParts p).body p.NC I hs 0 (Nat.zero_le _) s initial
  refine ⟨t,he,?_,hf.2.1,hf.2.2⟩
  simpa [Nat.add_mul,Nat.one_mul] using hf.1
#axiomsClean interQueryLoop
#axiomsClean interDeltaLoop
end
end VeriTile.Bench.Repairs.LightningAttention
