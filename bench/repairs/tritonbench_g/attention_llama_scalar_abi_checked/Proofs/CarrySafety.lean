import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarrySafetyStages
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryExecution

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
open Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Safety follows the same actual loop successors as the execution proof. -/
theorem range_safe (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ReachedBytes c s) (bounds : RegionName→Nat) (hb : AccessBounds c s bounds)
    (left n : Nat) (hn : n+left=count (stop c s))
    (t : State (Term α)) (h : Context c s scale n (accumulated c s scale n) t) :
    rangeSafe algebra bounds "start_n" (n*64) (stop c s) 64 (parts c).body t := by
  induction left generalizing n t with
  | zero =>
    have ht : ¬n*64<stop c s := by rw [count_test]; omega
    rw [rangeSafe_unfold]
    simp only [show (64:Nat)≠0 by decide,ht,↓reduceIte]
  | succ left ih =>
    have ht : n*64<stop c s := (count_test _ _).mpr (by omega)
    let next:=iteration_of_reached c s t scale n (accumulated c s scale n) h bytes
    have hc : Context c s scale (n+1) (accumulated c s scale (n+1)) next.val :=
      iteration_context_of_reached c s t scale n (accumulated c s scale n) h bytes
    rw [rangeSafe_unfold]
    simp only [show (64:Nat)≠0 by decide,ht,↓reduceIte]
    refine ⟨iteration_safe c s t scale n _ h bytes bounds hb,?_⟩
    rw [next.property]
    simpa only [Nat.add_mul,Nat.one_mul] using ih (n+1) (by omega) next.val hc

/-- Actual shared-source access safety for FP16 and E5M2, including preamble,
all dynamic iterations and final masked stores. -/
theorem source_safe (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ReachedBytes c s) (bounds : RegionName→Nat) (hb : AccessBounds c s bounds) :
    TraceSafe algebra bounds (kernel c) (bindScale s scale) := by
  obtain ⟨u,hu,hc⟩ := range_execution_of_reached c s scale bytes (count (stop c s)) 0 (by omega)
    (entry c s scale).val (entry_context c s scale)
  unfold TraceSafe
  rw [source_split,traceSafeList_append]
  refine ⟨entry_safe c s scale bounds hb,?_⟩
  intro t ht
  have he : t=(entry c s scale).val := Option.some.inj (ht.symm.trans (entry c s scale).property)
  subst t
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,entry_stop,Option.bind_some,true_and]
    exact range_safe c s scale bytes bounds hb (count (stop c s)) 0 (by omega)
      (entry c s scale).val (entry_context c s scale)
  · intro final ran
    have step_eq : step algebra
        (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat 64)
          (parts c).body) (entry c s scale).val=some u := by
      simpa only [step,evalOp_unfold,entry_stop,Option.bind_some,Nat.zero_mul] using hu
    have he : final=u := Option.some.inj (ran.symm.trans step_eq)
    subst final
    exact finish_safe c s u scale _ _ hc bounds hb

#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
