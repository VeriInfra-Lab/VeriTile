import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Specification

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

/-- Operand correspondence uses only initial memory reads and integer PID
coordinates. Successful execution and output equations are conclusions later. -/
theorem source_algorithm (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (read : RegionName→Nat→α) (block batch head : Nat)
    (hblock : s.pids 0=block) (hbatch : s.pids 1/p.heads=batch) (hhead : s.pids 1%p.heads=head)
    (hhz : s.pids 1=batch*p.heads+head)
    (loads : ∀ r∈[p.Q,p.K,p.V,p.QS,p.KS],∀ off,(s.mem r off).read .real=read r off) :
    answer M dot p s=Specification.algorithm M dot p read block batch head := by
  have loadQ:=loads p.Q (by simp)
  have loadK:=loads p.K (by simp)
  have loadV:=loads p.V (by simp)
  have loadQS:=loads p.QS (by simp)
  have loadKS:=loads p.KS (by simp)
  have q : Source.query M p s=Specification.query M p read block batch head := by
    funext i
    simp [Source.query,Specification.query,Source.zero,Specification.zero,loadQ,Source.base,hblock,hbatch,hhead,offset]
  have qs : Source.queryScale M p s=Specification.queryScale M p read block batch head := by
    simp [Source.queryScale,Specification.queryScale,loadQS,hblock,hhz]
  have k : ∀ row,Source.key M p s row=Specification.key M p read batch head row := by
    intro row
    funext i
    simp [Source.key,Specification.key,Source.zero,Specification.zero,loadK,Source.base,hbatch,hhead,offset,
      Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  have ks : ∀ row,Source.keyScale M p s row=Specification.keyScale M p read batch head row := by
    intro row
    simp [Source.keyScale,Specification.keyScale,loadKS,hhz]
  have v : ∀ row,Source.value M p s row=Specification.value M p read batch head row := by
    intro row
    funext i
    simp [Source.value,Specification.value,Source.zero,Specification.zero,loadV,Source.base,hbatch,hhead,offset]
  have scores : Source.scores M dot p s=Specification.scores M dot p read block batch head := by
    funext row i
    simp only [Source.scores,Specification.scores,q,qs,k,ks]
    rfl
  have maximum : Source.maximum M dot p s=Specification.maximum M dot p read block batch head := by
    funext row acc i
    simp only [Source.maximum,Specification.maximum,scores]
  have probability : Source.probability M dot p s=Specification.probability M dot p read block batch head := by
    funext row acc i
    simp only [Source.probability,Specification.probability,scores,maximum]
  have rescale : Source.rescale M dot p s=Specification.rescale M dot p read block batch head := by
    funext row acc i
    simp only [Source.rescale,Specification.rescale,maximum]
  have contribution : Source.contribution M dot p s=Specification.contribution M dot p read block batch head := by
    funext row acc i
    simp only [Source.contribution,Specification.contribution,probability,v]
    rfl
  have step : Source.update M dot p s=Specification.update M dot p read block batch head := by
    funext row acc
    simp only [Source.update,Specification.update,maximum,probability,rescale,contribution]
  unfold Whole.answer Source.outputValue Source.accumulated Specification.algorithm
  rw [step]
  rfl

theorem algorithm_at (M : Algebra α) (dot : Dot α) (c : Call) (read : RegionName→Nat→α)
    (b h n : Nat) (hh : h<(parameters c).heads) :
    answer M dot (parameters c)
      (start (grid (parameters c)) (n/128,b*(parameters c).heads+h,0) (entry read))=
      Specification.algorithm M dot (parameters c) read (n/128) b h := by
  have positive : 0<(parameters c).heads := by omega
  have div : (b*(parameters c).heads+h)/(parameters c).heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have mod : (b*(parameters c).heads+h)%(parameters c).heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  apply source_algorithm
  · simp [start,ids]
  · simpa [start,ids] using div
  · simpa [start,ids] using mod
  · simp [start,ids]
  · intro r hr off
    simp only [parameters,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl <;> simp [start,entry,tag]

#axiomsClean source_algorithm
#axiomsClean algorithm_at
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
