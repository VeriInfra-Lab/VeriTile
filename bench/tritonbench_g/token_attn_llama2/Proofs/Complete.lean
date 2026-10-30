import bench.tritonbench_g.token_attn_llama2.Proofs.GridRun

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Actual full quoted-source grid success, every active value, and complete
raw typed memory frame. There is no wrapper outcome or success premise. -/
theorem complete {α} [Inhabited α] (A : Program → Algebra α) (c : Call) (scale : ℝ)
    (m : Metadata c) (x : Buffers α c) (s : State α) (v : Valid c m)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    ∃ final, NativeGrid.runPrograms A (c.parameters scale) c.grid work (initial c m x s)=some final ∧
      Result c m (fun b h t => score (A (b,h,t/32)) (c.parameters scale)
        (start c.grid (b,h,t/32) (initial c m x s)) (lane c scale t)) (initial c m x s) final := by
  have uniq : Unique (c.parameters scale) c.grid work (initial c m x s) := by
    intro a ha b hb i j hi hj eq
    exact grid_unique c m x s scale v a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj eq
  obtain ⟨final,run,hv,hf⟩ := NativeGrid.run_correct A (c.parameters scale) (parameters_separated c scale) c.grid work (initial c m x s) uniq
  refine ⟨final,run,?_,?_⟩
  · intro b hb h hh t ht
    obtain ⟨hmem,hindex⟩ := grid_covers c m v b h t hb hh ht
    have hl := length_bound c m v b hb
    have ha : active (c.parameters scale) (start c.grid (b,h,t/32) (initial c m x s)) (lane c scale t) := by
      rw [initial_active]
      change c.maxLength-m.len b+(t/32*32+t%32)<c.maxLength
      omega
    have out := hv (b,h,t/32) (legal.mem_iff.mp hmem) (lane c scale t) ha
    rw [initial_offset] at out
    simpa only [lane,Prod.fst,Prod.snd,hindex] using out
  · intro r o hu
    apply hf
    rcases hu with hu | hu
    · exact Or.inl hu
    · right
      intro a ha i hi
      obtain ⟨hb,hh,_⟩ := (mem_programs c.grid a.1 a.2.1 a.2.2).mp (legal.mem_iff.mpr ha)
      have hl := length_bound c m v a.1 hb
      rw [initial_active] at hi
      rw [initial_offset]
      exact hu a.1 hb a.2.1 hh _ (by omega)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
