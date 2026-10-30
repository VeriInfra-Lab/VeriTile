import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetyPrologue
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetyBody
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetyEpilogue
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.AccessWindows

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (positive : 0<p.blockN)
    (backed : Accesses p bounds initial) (c : Nat) (current : State α) (hc : c≤p.blocks)
    (h : Invariant P p initial c (rounds P p initial c) current) :
    rangeSafe (backend A P) bounds "start_n" (c*p.blockN) p.length p.blockN (body p) current := by
  have aux : ∀ fuel c (current : State α),p.blocks-c=fuel → c≤p.blocks →
      Invariant P p initial c (rounds P p initial c) current →
      rangeSafe (backend A P) bounds "start_n" (c*p.blockN) p.length p.blockN (body p) current := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro c current he hc inv
      by_cases active : c<p.blocks
      · rw [rangeSafe,if_neg (by omega),if_pos ((block_guard p c positive).mpr active)]
        refine ⟨body_safe A P p bounds initial current c _ positive inv
          (backed.key c active) (backed.value c active) (backed.bias0 c active),?_⟩
        rw [bodyExecution A P p initial current c _ positive inv]
        have next := ih (p.blocks-(c+1)) (by omega) (c+1)
          (bodyResult P p initial current c (rounds P p initial c)) rfl (by omega)
          (bodyInvariant P p initial current c _ inv)
        simpa only [Nat.add_mul,Nat.one_mul] using next
      · rw [rangeSafe,if_neg (by omega),if_neg (by intro lt;exact active ((block_guard p c positive).mp lt))]
        trivial
  exact aux _ c current rfl hc h

theorem source_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (positive : 0<p.blockN)
    (backed : Accesses p bounds initial) :
    TraceSafe (backend A P) bounds (source p) initial := by
  change traceSafeList (backend A P) bounds (source p).surfaceBody initial
  rw [source_decomposition,traceSafeList_append,traceSafeList_append]
  refine ⟨⟨prologue_safe A P p bounds initial backed.query backed.bias1,?_⟩,?_⟩
  · intro first executed
    have same := (prologueExecution A P p initial).symm.trans executed
    cases Option.some.inj same
    rw [singleton_safe]
    have inv := prologueInvariant P p initial
    simpa [stepSafe_unfold,safeOp_unfold,evalOp_unfold,inv.lo,inv.hi]
      using loop_safe A P p bounds initial positive backed 0 _ (by omega) inv
  · intro middle executed
    obtain ⟨actual,run,inv⟩ := loopExecution A P p initial (prologueResult P p initial) 0 (by omega) positive
      (prologueInvariant P p initial)
    have loopRun : FP.Structural.run (backend A P)
        [.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat p.blockN) (body p)]
        (prologueResult P p initial)=some actual := by
      have inv₀ := prologueInvariant P p initial
      simpa [FP.Structural.run,step,evalOp_unfold,inv₀.lo,inv₀.hi] using run
    rw [run_append,prologueExecution,Option.bind_some,loopRun] at executed
    cases Option.some.inj executed
    exact epilogue_safe A P p bounds initial _ inv backed.output

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
