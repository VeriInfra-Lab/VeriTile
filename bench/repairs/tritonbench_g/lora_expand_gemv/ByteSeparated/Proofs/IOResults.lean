import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOResults
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 6000000

 theorem graph_cell (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (b n : Nat) :
    resultCell plans g raw b n=Cell.mk .real (graphResult plans g raw b n) := by
  unfold resultCell graphResult
  split <;> simp [entry]

theorem complete (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (order : List Nat)
    (all : order.Perm (List.range g.count)) (s : State Word)
    (hin : (io rest plans g order).InputCells (labeledInputs g raw) s) :
    ∃ final,runSchedule ((io rest plans g order).schedule (labeledInputs g raw)) s=some final ∧
      (io rest plans g order).OutputBinding (labeledInputs g raw)
        (graphTarget plans g raw (labeledInputs g raw)) final ∧
      (io rest plans g order).Frame (labeledInputs g raw) s final := by
  obtain ⟨final,hr,hf⟩ := IOGrid.execute_prefix rest plans g raw v s.mem order [] s
    (IOBinding.start rest plans g raw order s hin)
    (fun q hq => List.mem_range.mp (all.mem_iff.mp hq))
    (all.nodup_iff.mpr List.nodup_range) (by simp)
  refine ⟨final,hr,?_,?_⟩
  · intro o j _
    have hz : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,n,u⟩
    have hc := coverage g b.val n.val b.isLt n.isLt
    have hm : b.val*g.N+n.val∈order := all.mem_iff.mpr (List.mem_range.mpr hc.1)
    have hout : final.mem "Y" (yoff g b.val n.val)=resultCell plans g raw b.val n.val := by
      simpa [hm,offset,cell,hc.2.1,hc.2.2] using hf.2.1 (b.val*g.N+n.val) hc.1
    change mapCell id (final.mem "Y" (yoff g b.val n.val))=
      Cell.mk .real (graphResult plans g raw b.val n.val)
    rw [hout,graph_cell]
    rfl
  · intro r o outside
    apply hf.2.2 r o
    by_cases hy : r="Y"
    · right
      intro q hq he
      have hc := coordinates g q hq
      have hout := outside.1 (0 : Fin 1) (⟨q/g.N,hc.2⟩,⟨q%g.N,hc.1⟩,()) trivial
      change r≠("Y" : RegionName) ∨ o≠yoff g (q/g.N) (q%g.N) at hout
      exact (hout.resolve_left (by simpa [hy])).elim he
    · exact Or.inl hy

theorem real_target (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (data : RegionName → Nat → ℝ) (b n : Nat) :
    Real.eval data (graphResult plans g raw b n)=Real.target data g raw b n := by
  have h := Real.value_eq_target plans g raw v data b n
  rw [graph_cell,Cell.read_same] at h
  exact h

#axiomsClean complete
#axiomsClean real_target
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOResults
