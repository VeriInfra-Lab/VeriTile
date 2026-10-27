import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Triton.Float.Control
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

/-! Successful execution of all four canonical typed sources. Loop invariants
track only defined registers, not numerical identities or an output oracle.
The floating algebra remains opaque and total on every represented value. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.FPExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

@[simp] theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (mask : ι → Bool) (l : List ι) (s : State α) :
    (l.foldl (fun acc i => if mask i then acc.write r (off i) (val i) else acc) s).regs = s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    cases mask i <;> rfl

theorem integer_loop_invariant {α : Type} [Inhabited α] (M : Algebra α) {idx : RegName} {n : Nat} {body : List ComputeStmt}
    {P : Nat → State α → Prop}
    (hstep : ∀ i s, i < n → P i s →
      ∃ t, run M body (s.setReg idx .int [] (fun _ => (i : Int))) = some t ∧ P (i + 1) t)
    (start : Nat) (hstart : start ≤ n) (s : State α) (hs : P start s) :
    ∃ t, rangeInt M idx (start : Int) (n : Int) 1 body s = some t ∧ P n t := by
  have key : ∀ k i, n - i = k → i ≤ n → ∀ s, P i s →
      ∃ t, rangeInt M idx (i : Int) (n : Int) 1 body s = some t ∧ P n t := by
    intro k
    induction k with
    | zero =>
      intro i hk hi s hs
      have heq : i = n := by omega
      subst i
      exact ⟨s, by rw [rangeInt]; simp, hs⟩
    | succ k ih =>
      intro i hk hi s hs
      have hlt : i < n := by omega
      obtain ⟨t, ht, hp⟩ := hstep i s hlt hs
      obtain ⟨u, hu, hpu⟩ := ih (i + 1) (by omega) (by omega) t hp
      refine ⟨u, ?_, hpu⟩
      rw [rangeInt]
      simpa only [show (0 : Int) < 1 by decide, Int.ofNat_lt, hlt, and_self,
        if_true, ht, Option.bind_some, Int.natCast_add, Int.natCast_one] using hu
  exact key (n - start) start rfl hstart s hs

/-- Extract the actual source loop body; no second AST is maintained. -/
def bodyAt (kernel : ComputeKernel) (index : Nat) : List ComputeStmt :=
  match kernel.surfaceBody[index]? with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def forwardInvariant {α : Type} (length batch dim block : Nat) (u : State α) : Prop :=
  u.regs .int [] "length" = some (fun _ => (length : Int)) ∧
  u.regs .int [] "batch_size" = some (fun _ => (batch : Int)) ∧
  u.regs .int [] "dim" = some (fun _ => (dim : Int)) ∧
  ∃ v0 v1 v2 v3, u.regs .int [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "s" = some v2 ∧
    u.regs .real [block] "Lambda" = some v3

theorem forward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s x transition y : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.forwardTyped s x transition y length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.take 8) u = some a ∧
      forwardInvariant length batch dim block a := by
    simp [Kernels.forwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, forwardInvariant]
  have hs : ∀ i a, i < length → forwardInvariant length batch dim block a →
      ∃ b, run M (bodyAt (Kernels.forwardTyped s x transition y length batch dim block) 8)
        (a.setReg "t" .int [] (fun _ => (i : Int))) = some b ∧ forwardInvariant length batch dim block b := by
    intro i a _ hi
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, h0, h1, h2, h3⟩
    simp [bodyAt, Kernels.forwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, forwardInvariant,
      hLength, hBatch, hDim, h0, h1, h2, h3]
  have startLength := hi.1
  obtain ⟨b, hb, hi⟩ := integer_loop_invariant M (P := fun _ => forwardInvariant length batch dim block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.drop 9) b = some v := by
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, h0, h1, h2, h3⟩
    simp [Kernels.forwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      hLength, hBatch, hDim, h0, h1, h2, h3]
  have loopStep : step M (.forRangeInt "t" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.forwardTyped s x transition y length batch dim block) 8)) a = some b := by
    simp only [step, evalOp_unfold, startLength]
    exact hb
  refine ⟨v, ?_⟩
  have source : (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody =
      (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.take 8 ++
        [.forRangeInt "t" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.forwardTyped s x transition y length batch dim block) 8)] ++
          (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.drop 9 := rfl
  change run M (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, loopStep, Option.bind_some] using hv

def backwardInvariant {α : Type} (length batch dim block : Nat) (u : State α) : Prop :=
  u.regs .int [] "length" = some (fun _ => (length : Int)) ∧
  u.regs .int [] "batch_size" = some (fun _ => (batch : Int)) ∧
  u.regs .int [] "dim" = some (fun _ => (dim : Int)) ∧
  ∃ v0 v1 v2 v3 v4, u.regs .int [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "Lambda" = some v2 ∧
    u.regs .real [block] "grad_s" = some v3 ∧
    u.regs .real [block] "grad_Lambda" = some v4

theorem backward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 9) u = some a ∧
      backwardInvariant length batch dim block a := by
    simp [Kernels.backwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, backwardInvariant]
  have hs : ∀ i a, i < length → backwardInvariant length batch dim block a →
      ∃ b, run M (bodyAt (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) 9)
        (a.setReg "i" .int [] (fun _ => (i : Int))) = some b ∧ backwardInvariant length batch dim block b := by
    intro i a _ hi
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, h0, h1, h2, h3, h4⟩
    by_cases hbranch : (i : Int) < (length : Int) - 1 <;>
      simp [hbranch, bodyAt, Kernels.backwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, backwardInvariant,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4]
  have startLength := hi.1
  obtain ⟨b, hb, hi⟩ := integer_loop_invariant M (P := fun _ => backwardInvariant length batch dim block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 10) b = some v := by
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, h0, h1, h2, h3, h4⟩
    simp [Kernels.backwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4]
  have loopStep : step M (.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) 9)) a = some b := by
    simp only [step, evalOp_unfold, startLength]
    exact hb
  refine ⟨v, ?_⟩
  have source : (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody =
      (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 9 ++
        [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) 9)] ++
          (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 10 := rfl
  change run M (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, loopStep, Option.bind_some] using hv

def complexForwardInvariant {α : Type} (length batch dim block : Nat) (u : State α) : Prop :=
  u.regs .int [] "length" = some (fun _ => (length : Int)) ∧
  u.regs .int [] "batch_size" = some (fun _ => (batch : Int)) ∧
  u.regs .int [] "dim" = some (fun _ => (dim : Int)) ∧
  ∃ v0 v1 v2 v3 v4 v5, u.regs .int [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "s_real" = some v2 ∧
    u.regs .real [block] "s_imag" = some v3 ∧
    u.regs .real [block] "lambda_real" = some v4 ∧
    u.regs .real [block] "lambda_imag" = some v5

theorem complexForward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s x y transition : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.complexForwardTyped s x y transition length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.take 10) u = some a ∧
      complexForwardInvariant length batch dim block a := by
    simp [Kernels.complexForwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexForwardInvariant]
  have hs : ∀ i a, i < length → complexForwardInvariant length batch dim block a →
      ∃ b, run M (bodyAt (Kernels.complexForwardTyped s x y transition length batch dim block) 10)
        (a.setReg "t" .int [] (fun _ => (i : Int))) = some b ∧ complexForwardInvariant length batch dim block b := by
    intro i a _ hi
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, v5, h0, h1, h2, h3, h4, h5⟩
    simp [bodyAt, Kernels.complexForwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexForwardInvariant,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4, h5]
  have startLength := hi.1
  obtain ⟨b, hb, hi⟩ := integer_loop_invariant M (P := fun _ => complexForwardInvariant length batch dim block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.drop 11) b = some v := by
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, v5, h0, h1, h2, h3, h4, h5⟩
    simp [Kernels.complexForwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4, h5]
  have loopStep : step M (.forRangeInt "t" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.complexForwardTyped s x y transition length batch dim block) 10)) a = some b := by
    simp only [step, evalOp_unfold, startLength]
    exact hb
  refine ⟨v, ?_⟩
  have source : (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody =
      (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.take 10 ++
        [.forRangeInt "t" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.complexForwardTyped s x y transition length batch dim block) 10)] ++
          (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.drop 11 := rfl
  change run M (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, loopStep, Option.bind_some] using hv

def complexBackwardInvariant {α : Type} (length batch dim block : Nat) (u : State α) : Prop :=
  u.regs .int [] "length" = some (fun _ => (length : Int)) ∧
  u.regs .int [] "batch_size" = some (fun _ => (batch : Int)) ∧
  u.regs .int [] "dim" = some (fun _ => (dim : Int)) ∧
  ∃ v0 v1 v2 v3 v4 v5 v6 v7, u.regs .int [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "lambda_real" = some v2 ∧
    u.regs .real [block] "lambda_imag" = some v3 ∧
    u.regs .real [block] "grad_s_real" = some v4 ∧
    u.regs .real [block] "grad_s_imag" = some v5 ∧
    u.regs .real [block] "grad_lambda_real" = some v6 ∧
    u.regs .real [block] "grad_lambda_imag" = some v7

theorem complexBackward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 12) u = some a ∧
      complexBackwardInvariant length batch dim block a := by
    simp [Kernels.complexBackwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexBackwardInvariant]
  have hs : ∀ i a, i < length → complexBackwardInvariant length batch dim block a →
      ∃ b, run M (bodyAt (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) 12)
        (a.setReg "i" .int [] (fun _ => (i : Int))) = some b ∧ complexBackwardInvariant length batch dim block b := by
    intro i a _ hi
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, v5, v6, v7, h0, h1, h2, h3, h4, h5, h6, h7⟩
    by_cases hbranch : (i : Int) < (length : Int) - 1 <;>
      simp [hbranch, bodyAt, Kernels.complexBackwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexBackwardInvariant,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4, h5, h6, h7]
  have startLength := hi.1
  obtain ⟨b, hb, hi⟩ := integer_loop_invariant M (P := fun _ => complexBackwardInvariant length batch dim block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 13) b = some v := by
    rcases hi with ⟨hLength, hBatch, hDim, v0, v1, v2, v3, v4, v5, v6, v7, h0, h1, h2, h3, h4, h5, h6, h7⟩
    simp [Kernels.complexBackwardTyped, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      hLength, hBatch, hDim, h0, h1, h2, h3, h4, h5, h6, h7]
  have loopStep : step M (.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) 12)) a = some b := by
    simp only [step, evalOp_unfold, startLength]
    exact hb
  refine ⟨v, ?_⟩
  have source : (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody =
      (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 12 ++
        [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat 1) (bodyAt (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) 12)] ++
          (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 13 := rfl
  change run M (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, loopStep, Option.bind_some] using hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.FPExecution
