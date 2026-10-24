import bench.repairs.tritonbench_g.attention_llama.Proofs.RealKernel
import bench.repairs.tritonbench_g.attention_llama.Proofs.FP8Execution

namespace VeriTile.Bench.Repairs.AttentionLlama.FP8Kernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open FP8RealMemory
open FP8RealLoop (BytesFrom)
open LoopInvariant
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

noncomputable def view (c : Context) : Context :=
  { c with initial := decoded c.K c.V c.initial }

theorem execution (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hout : Postamble.ActiveInjective (c.initial.pids 0 * 64) c.H 64 c.D som son)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hundef : ∀ rg o, c.initial.undef rg o = 0)
    (hK : BytesFrom c.K c.initial (alBase c.initial c.NH c.skz c.skh) c.skn c.skk 0
      (FP8RealSetup.stop c.initial c.N c.SP 64 c.causal) c.N 64 c.D)
    (hV : BytesFrom c.V c.initial (alBase c.initial c.NH c.svz c.svh) c.svk c.svn 0
      (FP8RealSetup.stop c.initial c.N c.SP 64 c.causal) c.N 64 c.D) :
    ∃ sf, exec (TypedSource.repairedFP8 c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel c.initial = some sf ∧
      (∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        sf.readMem Out (alOutOffset c.initial c.NH soz soh som son 64 idx) =
          VeriTile.Bench.Repairs.AttentionLlama.RealKernel.target (view c) idx) ∧
      (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        o ≠ alOutOffset c.initial c.NH soz soh som son 64 idx) →
          sf.mem r o = c.initial.mem r o) := by
  obtain ⟨result,hrun,hread,hframe⟩ := VeriTile.Bench.Repairs.AttentionLlama.RealKernel.execution
    (view c) Out soz soh som son hD hN hSP hout hundef
  have he := FP8Execution.decoded_execution c.Q c.K c.V Out c.scale
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
    soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal c.initial hQK hQV hOK hOV hundef hK hV
  dsimp only [view] at hrun
  rw [hrun] at he
  cases hs : exec (TypedSource.repairedFP8 c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel c.initial with
  | none => simp [hs] at he
  | some sf =>
    rw [hs,Option.map_some,Option.some.injEq] at he
    subst result
    refine ⟨sf,rfl,?_,?_⟩
    · intro idx hi
      have hv := hread idx hi
      change (decoded c.K c.V sf).readMem Out _ = _ at hv
      unfold BlockState.readMem at hv ⊢
      rw [outside_memory c.K c.V sf Out _ hOK hOV] at hv
      exact hv
    · intro r o ho
      by_cases hr : r = c.K ∨ r = c.V
      · have hrO : r ≠ Out := by rcases hr with rfl | rfl <;> exact Ne.symm (by assumption)
        exact FP8Execution.memory_frame c.Q c.K c.V Out c.scale
          c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
          soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal c.initial sf r o hrO hundef hs
      · have hrK : r ≠ c.K := fun h => hr (Or.inl h)
        have hrV : r ≠ c.V := fun h => hr (Or.inr h)
        have hf := hframe r o ho
        change (decoded c.K c.V sf).mem r o = (decoded c.K c.V c.initial).mem r o at hf
        simpa only [outside_memory c.K c.V sf r o hrK hrV,
          outside_memory c.K c.V c.initial r o hrK hrV] using hf

end VeriTile.Bench.Repairs.AttentionLlama.FP8Kernel
