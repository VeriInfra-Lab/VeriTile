import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Host
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Real
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
/-- Ideal-real interpretation of the complete unchanged source, every legal
program schedule and the caller's original Y. Native rounding is separate. -/
theorem source_correct (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (data : RegionName → Nat → ℝ)
    (schedule : List Nat) (all : schedule.Perm (List.range g.count)) :
    ∃ final,execute rest plans g schedule (entry g raw)=some final ∧
      (∀ b n,b<g.B → n<g.N → eval data ((final "Y" (yoff g b n)).read .real)=target data g raw b n) ∧
      (∀ r o,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → o≠yoff g b n) →
        final r o=entry g raw r o) := by
  obtain ⟨final,he,hv,hf⟩ := whole_source rest plans g raw v schedule all
  refine ⟨final,he,?_,hf⟩
  intro b n hb hn
  rw [hv b n hb hn,value_eq_target plans g raw v data b n]

/-- Complete original host dispatch and supplied-output API, interpreted over
real arithmetic. The native bit theorem is separate and retains conversions. -/
theorem correct (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (call : Host.Preparation g) (data : RegionName → Nat → ℝ)
    (schedule : List Nat) (all : schedule.Perm (List.range g.count)) :
    ∃ result,Host.invoke rest plans g call schedule=some result ∧ result.returned=none ∧
      result.outputIdentity=call.outputIdentity ∧ result.callerDevice=call.callerDevice ∧
      result.grid=(g.N,g.B) ∧
      (∀ b n,b<g.B → n<g.N → eval data ((result.memory "Y" (yoff g b n)).read .real)=target data g call.raw b n) ∧
      (∀ r o,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → o≠yoff g b n) →
        result.memory r o=entry g call.raw r o) := by
  obtain ⟨result,he,hn,ho,hd,hgrid,hv,hf⟩ := Host.whole_call rest plans g call schedule all
  refine ⟨result,he,hn,ho,hd,hgrid,?_,hf⟩
  intro b n hb hn
  rw [hv b n hb hn,value_eq_target plans g call.raw call.valid data b n]

#axiomsClean source_correct
#axiomsClean correct
#specNonCircular target avoiding [Host.invoke,execute,Branches.source,FP.Structural.exec,value]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Real
