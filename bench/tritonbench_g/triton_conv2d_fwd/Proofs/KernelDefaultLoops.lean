import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultExecution
namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem Context.set_loop {α : Type} (p : Parameters) (initial current : State α)
    (ctx : Context p initial current) (idx : RegName) (n : Nat)
    (hi : idx="h" ∨ idx="w" ∨ idx="c") :
    Context p initial (current.setReg idx .nat [] (fun _ => n)) := by
  rcases hi with rfl | rfl | rfl
  all_goals constructor <;> simp [ctx.memory, ctx.undefined, ctx.group, ctx.ig,
    ctx.og, ctx.ba, ctx.fe, ctx.he, ctx.wi, ctx.ip, ctx.wp]

@[simp] theorem hasAcc_set_nat {α : Type} (p : Parameters) (a : CIndex p → α)
    (s : State α) (idx : RegName) (n : Nat) (hi : idx ≠ "accum") :
    HasAcc p a (s.setReg idx .nat [] (fun _ => n)) ↔ HasAcc p a s := by
  simp [HasAcc, State.setReg, Ne.symm hi]

theorem channelLoop {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (h w : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (hw : current.regs .nat [] "w" = some (fun _ => w)) :
    ∃ t, run (backend M D) (wBody p) current = some t ∧
      Context p initial t ∧ HasAcc p (channels M D p initial h w a) t ∧
      t.regs .nat [] "h" = some (fun _ => h) := by
  let I := fun (_ : Nat) (a : CIndex p → α) (t : State α) =>
    Context p initial t ∧ HasAcc p a t ∧
      t.regs .nat [] "h" = some (fun _ => h) ∧
      t.regs .nat [] "w" = some (fun _ => w)
  obtain ⟨t, ht, hc, ha', hh', hw'⟩ := range_values (backend M D) "c" (cBody p)
    (update M D p initial h w) I (fun a t => I 0 a t) 0 (igd p) (BIN p) a current
    (by
      intro n a t hi hn
      let next := bodyExecution M D p initial t h w n a hi.1 hi.2.1 hi.2.2.1 hi.2.2.2
      exact ⟨next.val, next.property, bodyContext M D p initial t h w n a hi.1 hi.2.1 hi.2.2.1 hi.2.2.2⟩)
    (by intro n a t hi; exact hi) ⟨ctx,ha,hh,hw⟩
  refine ⟨t, ?_, hc, ha', hh'⟩
  rw [wBody_shape]
  simpa [run, step, evalOp_unfold, ctx.ig] using ht

theorem widthLoop {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (h : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h)) :
    ∃ t, run (backend M D) (hBody p) current = some t ∧
      Context p initial t ∧ HasAcc p (widths M D p initial h a) t := by
  let I := fun (_ : Nat) (a : CIndex p → α) (t : State α) =>
    Context p initial t ∧ HasAcc p a t ∧ t.regs .nat [] "h" = some (fun _ => h)
  obtain ⟨t, ht, hc, ha', hh'⟩ := range_values (backend M D) "w" (wBody p)
    (channels M D p initial h) I (fun a t => I 0 a t) 0 p.kernel_width 1 a current
    (by
      intro w a t hi hn
      obtain ⟨u, he, hc, ha, hh⟩ := channelLoop M D p initial
        (t.setReg "w" .nat [] (fun _ => w)) h w a
        (hi.1.set_loop p initial t "w" w (Or.inr (Or.inl rfl)))
        (by simpa using hi.2.1) (by simpa using hi.2.2) (by simp)
      exact ⟨u,he,hc,ha,hh⟩)
    (by intro n a t hi; exact hi) ⟨ctx,ha,hh⟩
  refine ⟨t, ?_, hc, ha'⟩
  rw [hBody_shape]
  simpa [run, step] using ht

theorem heightLoop {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    ∃ t, run (backend M D) [.forRange "h" 0 p.kernel_height 1 (hBody p)] current = some t ∧
      Context p initial t ∧
      HasAcc p (foldRange (widths M D p initial) 0 p.kernel_height 1 a) t := by
  let I := fun (_ : Nat) (a : CIndex p → α) (t : State α) => Context p initial t ∧ HasAcc p a t
  obtain ⟨t, ht, hc, ha'⟩ := range_values (backend M D) "h" (hBody p)
    (widths M D p initial) I (fun a t => I 0 a t) 0 p.kernel_height 1 a current
    (by
      intro h a t hi hn
      obtain ⟨u,he,hc,ha⟩ := widthLoop M D p initial (t.setReg "h" .nat [] (fun _ => h)) h a
        (hi.1.set_loop p initial t "h" h (Or.inl rfl)) (by simpa using hi.2) (by simp)
      exact ⟨u,he,hc,ha⟩)
    (by intro n a t hi; exact hi) ⟨ctx,ha⟩
  exact ⟨t, by simpa [run, step] using ht, hc, ha'⟩

#axiomsClean channelLoop
#axiomsClean widthLoop
#axiomsClean heightLoop
end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault
