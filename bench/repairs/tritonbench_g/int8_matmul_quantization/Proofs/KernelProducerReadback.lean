import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducerResult

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
noncomputable section

theorem outputOffset_injective (p : Parameters) (s : State α)
    (stride : p.outputColumnStride = 1) : Function.Injective (outputOffset p s) := by
  intro i j h
  have hc : i.2.1 = j.2.1 := Fin.ext (by simpa [outputOffset, stride] using h)
  exact Prod.ext (Subsingleton.elim _ _) (Prod.ext hc (Subsingleton.elim _ _))

def Outside (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r ≠ p.scales ∨ o ≠ s.pids 0*p.scaleStride) ∧
    (r ≠ p.output ∨ ∀ i : Index p, i.2.1.val < p.columns → o ≠ outputOffset p s i)

/-- Whole quoted producer: every active INT8 lane, the actual FP16 scale
store, and the exterior memory frame. The stride/separation facts describe
the ordinary contiguous output allocation, not new input restrictions. -/
theorem source_correct {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) (width : 0 < p.width)
    (stride : p.outputColumnStride = 1) (regions : p.output ≠ p.scales) :
    ∃ final, FP.Structural.exec (backend M integer8) (KernelSource.quantize (parameters p)) s = some final ∧
      (∀ i : Index p, i.2.1.val < p.columns →
        final.mem p.output (outputOffset p s i) = .mk .int (integer8 (quotient M p s i)).toInt) ∧
      final.mem p.scales (s.pids 0*p.scaleStride) = .mk .fp16 (M.cast none .real .fp16 (scale M p s)) ∧
      (∀ r o, Outside p s r o → final.mem r o = s.mem r o) := by
  let finish := execution M integer8 p s width
  have hm := execution_memory M integer8 p s width
  refine ⟨finish.val, finish.property, ?_, ?_, ?_⟩
  · intro i hi
    rw [hm]
    simp only [reference, State.write, regions, false_and, if_false]
    exact State.masked_scatter_readback p.output (outputOffset p s) _
      (fun i => i.2.1.val < p.columns) s i hi (outputOffset_injective p s stride)
  · rw [hm]
    simp [reference, State.write]
  · intro r o h
    rw [hm]
    have hnot : ¬ (r = p.scales ∧ o = s.pids 0*p.scaleStride) := by
      rcases h.1 with hr | ho
      · exact fun hx => hr hx.1
      · exact fun hx => ho hx.2
    simp only [reference, State.write, hnot, if_false]
    unfold rowWritten
    apply State.masked_scatter_frame
    rcases h.2 with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
