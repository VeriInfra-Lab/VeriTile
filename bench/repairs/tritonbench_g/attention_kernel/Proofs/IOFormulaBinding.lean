import bench.repairs.tritonbench_g.attention_kernel.Target
import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Binding

/-! Only finite input windows enter the source recurrence. No condition is
imposed on unused memory, padding cells, or inactive query lanes. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.IOFormulaBinding
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open KernelOnly
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Reads (p : Parameters) (read : RegionName → Nat → α)
    (batch head : Nat) (inputs : Target.Views α) : Prop where
  query : ∀ r, r<p.length → ∀ d, d<p.dim →
    read p.Q (offset p.q batch head r d)=inputs.q r d
  key : ∀ r, r<p.length → ∀ d, d<p.dim →
    read p.K (offset p.k batch head r d)=inputs.k r d
  value : ∀ r, r<p.length → ∀ d, d<p.dim →
    read p.V (offset p.v batch head r d)=inputs.v r d
  bias : ∀ r, r<p.length → ∀ c, c<128 →
    read p.Bias (batch*p.bias.batch+head*p.bias.head+r*p.bias.row+c)=inputs.bias r c

theorem update_eq (M : Algebra α) (dot : Dot α) (p : Parameters)
    (read : RegionName → Nat → α) (block batch head : Nat) (inputs : Target.Views α)
    (loads : Reads p read batch head inputs) (guard : Guard p) (row : Nat) (hr : row<p.length) :
    Specification.update M dot p read block batch head row=Target.Native.update M dot p inputs block row := by
  have q : Specification.query M p read block batch head=Target.Native.query M p inputs block := by
    funext i
    by_cases hi : block*p.bm+i.1.val<p.length
    · simp [Specification.query,Target.Native.query,hi,loads.query _ hi _ i.2.1.isLt]
    · simp [Specification.query,Target.Native.query,hi]
  have k : Specification.key M p read block batch head row=Target.Native.key M p inputs block row := by
    funext i
    by_cases hi : row+i.2.1.val<p.length
    · simp [Specification.key,Target.Native.key,hi,loads.key _ hi _ i.1.isLt]
    · simp [Specification.key,Target.Native.key,hi]
  have v : Specification.value M p read block batch head row=Target.Native.value M p inputs block row := by
    funext i
    by_cases hi : row+i.1.val<p.length
    · simp [Specification.value,Target.Native.value,hi,loads.value _ hi _ i.2.1.isLt]
    · simp [Specification.value,Target.Native.value,hi]
  have b1 : Specification.biasStatic M p read block batch head=Target.Native.biasStatic M p inputs block := by
    funext i
    by_cases hi : block*p.bm+i.1.val<p.length
    · have hc : i.2.1.val%64+64<128 := by omega
      simp [Specification.biasStatic,Target.Native.biasStatic,hi,loads.bias _ hi _ hc]
    · simp [Specification.biasStatic,Target.Native.biasStatic,hi]
  have b0 : Specification.biasDynamic M p read block batch head row=Target.Native.biasDynamic M p inputs block row := by
    funext i
    by_cases hi : block*p.bm+i.val<p.length
    · simp [Specification.biasDynamic,Target.Native.biasDynamic,hi,
        loads.bias _ hi _ (bias_index_bound p guard row hr)]
    · simp [Specification.biasDynamic,Target.Native.biasDynamic,hi]
  have scores : Specification.scores M dot p read block batch head row=Target.Native.scores M dot p inputs block row := by
    funext i
    simp only [Specification.scores,Target.Native.scores,q,k,b0,b1,Specification.zero,Target.Native.zero]
  have maximum : Specification.maximum M dot p read block batch head row=Target.Native.maximum M dot p inputs block row := by
    funext a i
    simp only [Specification.maximum,Target.Native.maximum,scores]
  have probability : Specification.probability M dot p read block batch head row=Target.Native.probability M dot p inputs block row := by
    funext a i
    simp only [Specification.probability,Target.Native.probability,scores,maximum]
  have rescale : Specification.rescale M dot p read block batch head row=Target.Native.rescale M dot p inputs block row := by
    funext a i
    simp only [Specification.rescale,Target.Native.rescale,maximum]
  have contribution : Specification.contribution M dot p read block batch head row=Target.Native.contribution M dot p inputs block row := by
    funext a i
    simp only [Specification.contribution,Target.Native.contribution,probability,v]
  funext a
  simp only [Specification.update,Target.Native.update,maximum,rescale,probability,contribution]

theorem fold_congr {β : Type} (f g : Nat → β → β) (cur stop stride : Nat) (a : β)
    (h : ∀ r, r<stop → f r=g r) :
    foldRange f cur stop stride a=foldRange g cur stop stride a := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  by_cases hz : stride=0
  · simp [hz]
  · simp only [hz,↓reduceDIte]
    by_cases hc : cur<stop
    · simp only [hc,↓reduceDIte,h cur hc]
      exact fold_congr f g (cur+stride) stop stride (g cur a) h
    · simp [hc]
termination_by stop-cur
decreasing_by omega

theorem algorithm_eq (M : Algebra α) (dot : Dot α) (p : Parameters)
    (read : RegionName → Nat → α) (block batch head : Nat) (inputs : Target.Views α)
    (loads : Reads p read batch head inputs) (guard : Guard p) :
    Specification.algorithm M dot p read block batch head=Target.Native.algorithm M dot p inputs block := by
  have folded:=fold_congr (Specification.update M dot p read block batch head)
    (Target.Native.update M dot p inputs block) 0 p.length p.bn (Target.Native.initial M p)
    (fun row hr=>update_eq M dot p read block batch head inputs loads guard row hr)
  funext i
  simpa only [Specification.algorithm,Target.Native.algorithm,Specification.initial,Target.Native.initial,
    Specification.zero,Target.Native.zero] using congrArg
    (fun a : Acc p α=>M.cast (some .fp32) .real p.storage.float
      (M.binary (some .fp32) .real .div (a.value i) (a.normalizer i.1))) folded

def readState (p : Parameters) (s : State α) (r : RegionName) (off : Nat) : α :=
  toFloat p.storage.float ((s.mem r off).read p.storage.tile)

theorem source_at (M : Algebra α) (dot : Dot α) (p : Parameters) (s : State α)
    (block batch head : Nat) (hh : head<p.heads) (inputs : Target.Views α)
    (loads : Reads p (readState p s) batch head inputs) (guard : Guard p) :
    answer M dot p (start (grid p) (block,batch*p.heads+head,0) s)=
      Target.Native.algorithm M dot p inputs block := by
  have positive : 0<p.heads := by omega
  have div : (batch*p.heads+head)/p.heads=batch := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh,Nat.zero_add]
  have mod : (batch*p.heads+head)%p.heads=head := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  rw [KernelOnly.source_algorithm M dot p _ (readState p s) block batch head
    (by simp [start,ids]) (by simpa [start,ids] using div) (by simpa [start,ids] using mod)
    (by intros; rfl)]
  exact algorithm_eq M dot p _ block batch head inputs loads guard

#axiomsClean source_at
end
end VeriTile.Bench.Repairs.AttentionKernel.IOFormulaBinding
