import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativeIO

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def Agree (scope : RegionName → Nat → Prop) (a b : Memory α) : Prop :=
  ∀ r o, scope r o → a r o = b r o

theorem fold_rel {β γ : Type} (R : β → γ → Prop)
    (f : Nat → β → β) (g : Nat → γ → γ) (n : Nat) (a : β) (b : γ)
    (initial : R a b) (step : ∀ j x y, R x y → R (f j x) (g j y)) :
    R (foldRange f 0 n 1 a) (foldRange g 0 n 1 b) := by
  induction n with
  | zero => simpa only [foldRange_zero] using initial
  | succ n ih => simpa only [foldRange_succ] using step n _ _ ih

theorem scatter_agree {ι : Type} (scope : RegionName → Nat → Prop)
    (xs : List ι) (dest : RegionName) (active : ι → Prop) [DecidablePred active]
    (offset : ι → Nat) (value : ι → Cell α) (a b : Memory α) (h : Agree scope a b) :
    Agree scope (scatter xs dest active offset value a) (scatter xs dest active offset value b) := by
  induction xs generalizing a b with
  | nil => exact h
  | cons i rest ih =>
    apply ih
    intro r o hr
    by_cases ha : active i
    · simp only [if_pos ha]
      by_cases he : r=dest ∧ o=offset i
      · simp only [if_pos he]
      · simpa only [if_neg he] using h r o hr
    · simpa only [if_neg ha] using h r o hr

theorem read_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (a b : Memory α) (h : Agree scope a b) (r : RegionName) (o : Nat)
    (active : Prop) [Decidable active] (covered : active → scope r o) :
    read M a r o active = read M b r o active := by
  unfold read
  split
  · rename_i ha; rw [h r o (covered ha)]
  · rfl

structure ForwardCovered (scope : RegionName → Nat → Prop)
    (p : Parameters) (pid : PID) : Prop where
  query : ∀ t a, t<p.N → a<p.D → scope "query" (pid.head*p.N*p.D+t*p.D+a)
  key : ∀ t a, t<p.N → a<p.D → scope "key" (pid.head*p.N*p.D+t*p.D+a)
  value : ∀ t c, t<p.N → c<p.BM →
    scope "value" (pid.head*p.N*p.E+t*p.E+(pid.channel*p.BM+c))

theorem fwdQ_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    fwdQ M p pid a block = fwdQ M p pid b block := by
  funext i
  apply read_agree scope M a b h
  intro hi
  simpa only [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hc.query _ _ hi i.2.1.isLt

theorem fwdK_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    fwdK M p pid a block = fwdK M p pid b block := by
  funext i
  apply read_agree scope M a b h
  intro hi
  simpa only [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hc.key _ _ hi i.1.isLt

theorem fwdV_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    fwdV M p pid a block = fwdV M p pid b block := by
  funext i
  apply read_agree scope M a b h
  intro hi
  simpa only [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hc.value _ _ hi i.2.1.isLt

theorem fwdCarry_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    fwdCarry M D p pid a block = fwdCarry M D p pid b block := by
  induction block with
  | zero => rfl
  | succ block ih =>
    simp only [fwdCarry,fwdNext,ih,fwdK_agree scope M p pid hc a b h,
      fwdV_agree scope M p pid hc a b h]

theorem fwdOutput_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    fwdOutput M D p pid a block (fwdCarry M D p pid a block) =
      fwdOutput M D p pid b block (fwdCarry M D p pid b block) := by
  simp only [fwdOutput,fwdScore,fwdQ_agree scope M p pid hc a b h,
    fwdK_agree scope M p pid hc a b h,fwdV_agree scope M p pid hc a b h,
    fwdCarry_agree scope M D p pid hc a b h]

theorem fwdMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (pid : PID) (hc : ForwardCovered scope p pid)
    (a b x y : Memory α) (h : Agree scope a b) (hxy : Agree scope x y) :
    Agree scope (fwdMemory M D p pid a x) (fwdMemory M D p pid b y) := by
  apply fold_rel (Agree scope) _ _ _ _ _ hxy
  intro j u v huv
  rw [fwdOutput_agree scope M D p pid hc a b h j]
  exact scatter_agree scope _ _ _ _ _ u v huv

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality
