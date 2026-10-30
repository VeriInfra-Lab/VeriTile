import bench.tritonbench_g.attention_llama.Proofs.Loop
import bench.tritonbench_g.attention_llama.Proofs.CarryIterationContext
import bench.tritonbench_g.attention_llama.Proofs.Geometry

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem range_execution_of_reached (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ReachedBytes c s) (left n : Nat) (hn : n+left=count (stop c s))
    (t : State (Term α)) (h : Context c s scale n (accumulated c s scale n) t) :
    ∃ u, range algebra "start_n" (n*64) (stop c s) 64 (parts c).body t=some u ∧
      Context c s scale (count (stop c s)) (accumulated c s scale (count (stop c s))) u := by
  induction left generalizing n t with
  | zero =>
    have he : n=count (stop c s) := by omega
    have ht : ¬n*64<stop c s := by rw [count_test]; omega
    refine ⟨t,?_,?_⟩
    · rw [range]; simp only [show (64:Nat)≠0 by decide,ht,↓reduceIte]
    · simpa only [he] using h
  | succ left ih =>
    have ht : n*64<stop c s := (count_test _ _).mpr (by omega)
    let next := iteration_of_reached c s t scale n (accumulated c s scale n) h bytes
    have hc : Context c s scale (n+1) (accumulated c s scale (n+1)) next.val :=
      iteration_context_of_reached c s t scale n (accumulated c s scale n) h bytes
    obtain ⟨u,hu,hc'⟩ := ih (n+1) (by omega) next.val hc
    refine ⟨u,?_,hc'⟩
    rw [range]
    simp only [show (64:Nat)≠0 by decide,ht,↓reduceIte,next.property,Option.bind_some]
    simpa only [Nat.add_mul,Nat.one_mul] using hu

theorem range_execution (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ByteRange s) (left n : Nat) (hn : n+left=count (stop c s))
    (t : State (Term α)) (h : Context c s scale n (accumulated c s scale n) t) :
    ∃ u, range algebra "start_n" (n*64) (stop c s) 64 (parts c).body t=some u ∧
      Context c s scale (count (stop c s)) (accumulated c s scale (count (stop c s))) u :=
  range_execution_of_reached c s scale (bytes.reached c s) left n hn t h

theorem source_split (c : Config) :
    (kernel c).surfaceBody=(parts c).before++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat 64)
        (parts c).body::(parts c).after := by
  cases hf : c.fp8 <;> cases hc : c.causal <;>
    simp [parts,kernel,hf,hc,TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD]

theorem entry_stop (c : Config) (s : State (Term α)) (scale : Term α) :
    (entry c s scale).val.regs .nat [] "block_n_end"=some (fun _=>stop c s) := by
  rcases c with ⟨Q,K,V,Out,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svk,svn,soz,soh,som,son,NH,H,N,SP,D,causal,fp8⟩
  cases fp8 <;> cases causal <;>
    simp [entry,Source.stop,MaskCoverage.stop,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

#axiomsClean range_execution_of_reached
#axiomsClean range_execution
#axiomsClean source_split
#axiomsClean entry_stop
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
