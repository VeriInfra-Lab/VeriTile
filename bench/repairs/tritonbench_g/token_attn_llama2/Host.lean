import bench.repairs.tritonbench_g.token_attn_llama2.Domain
import bench.repairs.tritonbench_g.token_attn_llama2.NativeGrid
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
structure Returned (α : Type) where
  state : State α
  output : Descriptor
  returned : Option Unit

/-- Successful selected host path: exact grid, caller output descriptor, None.
Each CTA reads the preceding CTA's current memory. -/
def invoke (A : Program → Algebra α) (c : Call) (scale : ℝ) (abi : ABI c)
    (work : List Program) (s : State α) : Option (Returned α) :=
  (NativeGrid.runPrograms A (c.parameters scale) c.grid work s).map fun t => ⟨t,abi.tensor .output,none⟩

theorem host_result_with (A : Program → Algebra α) (c : Call) (scale : ℝ)
    (m : Metadata c) (x : Buffers α c) (s : State α) (v : Valid c m)
    (abi : ABI c) (work : List Program) (legal : (programs c.grid).Perm work) :
    ∃ result, invoke A c scale abi work (initial c m x s)=some result ∧
      result.returned=none ∧ result.output=abi.tensor .output ∧
      Result c m (fun b h t => score (A (b,h,t/32)) (c.parameters scale)
        (start c.grid (b,h,t/32) (initial c m x s)) (lane c scale t)) (initial c m x s) result.state := by
  have uniq : Unique (c.parameters scale) c.grid work (initial c m x s) := by
    intro a ha b hb i j hi hj eq
    exact grid_unique c m x s scale v a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj eq
  obtain ⟨final,run,hv,hf⟩ := NativeGrid.run_correct A (c.parameters scale) (parameters_separated c scale) c.grid work (initial c m x s) uniq
  refine ⟨⟨final,abi.tensor .output,none⟩,by simp [invoke,run],rfl,rfl,?_,?_⟩
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

#axiomsClean host_result_with
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
