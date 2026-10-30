import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.ReadProjection

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The actual source runs on the caller's regional state. Its read-only
projection equals the retained physical source step under internal view IDs;
the exact original cell frame is proved before taking any decoded projection. -/
theorem program (M : Algebra α) (ops : Operations α) (law : ScalarLaws M ops)
    (p : Problem) (b : Buffers) (L : Layout p b) (s : State α) (ht : s.pids 0<p.tiles) :
    ∃ final, FP.Structural.exec (backend M (selectedDot ops)) (source p b) s=some final ∧
      decode b final=Source.step (sourceArithmetic ops) p (views b) (s.pids 0) (decode b s) ∧
      (∀ (r : RegionName) o, (r≠"C" ∨ ∀ i, active p s i → o≠cOffset p b s i) →
        final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,_,value,frame⟩ := programExecution M (selectedDot ops) p b s
    (active_injective p b L s ht)
  let n := normalized b s
  have hn : n.pids 0<p.tiles := by simpa [n] using ht
  obtain ⟨canonical,crun,cp,_⟩ := program_projection M ops law p (views b) (layout p b L)
    (decode b s) (s.pids 0) ht s
  obtain ⟨other,orun,_,cv,cf⟩ := programExecution M (selectedDot ops) p (views b) n
    (active_injective p (views b) (layout p b L) n hn)
  have eq : other=canonical := Option.some.inj (orun.symm.trans crun)
  subst other
  have ve : outputValue M (selectedDot ops) p b s=
      outputValue M (selectedDot ops) p (views b) n := by
    exact output_congr M (selectedDot ops) p b s n (by simp [n])
      (fun o => (normalized_a b s o).symm) (fun o => (normalized_b b s o).symm)
  have cc : ∀ o, (final.mem "C" o).read .fp16=(canonical.mem "C" o).read .fp16 := by
    intro o
    by_cases written : ∃ i, active p s i ∧ o=cOffset p b s i
    · obtain ⟨i,hi,rfl⟩ := written
      have hi' : active p n i := hi
      have eq := cv i hi'
      change canonical.mem "C" (cOffset p b s i)=
        Cell.mk .fp16 (outputValue M (selectedDot ops) p (views b) n i) at eq
      rw [value i hi,eq,ve]
    · have outside : ∀ i, active p s i → o≠cOffset p b s i := by
        intro i hi eq; exact written ⟨i,hi,eq⟩
      rw [frame "C" o (Or.inr outside),cf "C" o (Or.inr outside)]
      exact (normalized_c b s o).symm
  have fa (o : Nat) : final.mem "A" o=s.mem "A" o := frame "A" o (Or.inl (by decide))
  have fb (o : Nat) : final.mem "B" o=s.mem "B" o := frame "B" o (Or.inl (by decide))
  refine ⟨final,ran,?_,frame⟩
  rw [←cp]
  funext point
  obtain ⟨r,o⟩ := point
  by_cases a : r=0
  · subst r; simp [decode,project,views,fa]
  by_cases bb : r=1
  · subst r; simp [decode,project,views,fb]
  by_cases c : r=2
  · subst r
    by_cases bound : b.C.offset≤o <;> simp [decode,project,views,cc,bound]
  simp [decode,project,views,a,bb,c]

#axiomsClean program
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReadProjection
