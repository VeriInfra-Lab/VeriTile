import bench.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.tritonbench_g.diag_ssm_triton.Kernels
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

/-- Extract the actual source loop body; no second AST is maintained. -/
def bodyAt (kernel : ComputeKernel) (index : Nat) : List ComputeStmt :=
  match kernel.surfaceBody[index]? with
  | some (.forLoop _ _ body) => body
  | _ => []

def forwardInvariant {α : Type} (block : Nat) (u : State α) : Prop :=
  ∃ v0 v1 v2 v3, u.regs .nat [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "s" = some v2 ∧
    u.regs .real [block] "Lambda" = some v3

theorem forward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s x transition y : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.forwardTyped s x transition y length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.take 5) u = some a ∧
      forwardInvariant block a := by
    simp [Kernels.ReadableSource.forward_preserved, TypedSource.forward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, forwardInvariant]
  have hs : ∀ i a, i < length → forwardInvariant block a →
      ∃ b, run M (bodyAt (Kernels.forwardTyped s x transition y length batch dim block) 5)
        (a.setReg "t" .nat [] (fun _ => i)) = some b ∧ forwardInvariant block b := by
    intro i a _ hi
    rcases hi with ⟨v0, v1, v2, v3, h0, h1, h2, h3⟩
    simp [bodyAt, Kernels.ReadableSource.forward_preserved, TypedSource.forward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, forwardInvariant,
      h0, h1, h2, h3]
  obtain ⟨b, hb, hi⟩ := loop_invariant M (P := fun _ => forwardInvariant block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.drop 6) b = some v := by
    rcases hi with ⟨v0, v1, v2, v3, h0, h1, h2, h3⟩
    simp [Kernels.ReadableSource.forward_preserved, TypedSource.forward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      h0, h1, h2, h3]
  refine ⟨v, ?_⟩
  have source : (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody =
      (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.take 5 ++
        [.forLoop "t" length (bodyAt (Kernels.forwardTyped s x transition y length batch dim block) 5)] ++
          (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody.drop 6 := rfl
  change run M (Kernels.forwardTyped s x transition y length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, step, Option.bind_some, hb] using hv

def backwardInvariant {α : Type} (block : Nat) (u : State α) : Prop :=
  ∃ v0 v1 v2 v3 v4, u.regs .nat [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "Lambda" = some v2 ∧
    u.regs .real [block] "grad_s" = some v3 ∧
    u.regs .real [block] "grad_Lambda" = some v4

theorem backward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 6) u = some a ∧
      backwardInvariant block a := by
    simp [Kernels.ReadableSource.backward_preserved, TypedSource.backward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, backwardInvariant]
  have hs : ∀ i a, i < length → backwardInvariant block a →
      ∃ b, run M (bodyAt (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) 6)
        (a.setReg "i" .nat [] (fun _ => i)) = some b ∧ backwardInvariant block b := by
    intro i a _ hi
    rcases hi with ⟨v0, v1, v2, v3, v4, h0, h1, h2, h3, h4⟩
    by_cases hbranch : i < length - 1 <;>
      simp [hbranch, bodyAt, Kernels.ReadableSource.backward_preserved, TypedSource.backward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, backwardInvariant,
      h0, h1, h2, h3, h4]
  obtain ⟨b, hb, hi⟩ := loop_invariant M (P := fun _ => backwardInvariant block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 7) b = some v := by
    rcases hi with ⟨v0, v1, v2, v3, v4, h0, h1, h2, h3, h4⟩
    simp [Kernels.ReadableSource.backward_preserved, TypedSource.backward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      h0, h1, h2, h3, h4]
  refine ⟨v, ?_⟩
  have source : (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody =
      (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 6 ++
        [.forLoop "i" length (bodyAt (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block) 6)] ++
          (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 7 := rfl
  change run M (Kernels.backwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, step, Option.bind_some, hb] using hv

def complexForwardInvariant {α : Type} (block : Nat) (u : State α) : Prop :=
  ∃ v0 v1 v2 v3 v4 v5, u.regs .nat [block] "col_offsets" = some v0 ∧
    u.regs .bool [block] "mask" = some v1 ∧
    u.regs .real [block] "s_real" = some v2 ∧
    u.regs .real [block] "s_imag" = some v3 ∧
    u.regs .real [block] "lambda_real" = some v4 ∧
    u.regs .real [block] "lambda_imag" = some v5

theorem complexForward_success {α : Type} [Inhabited α] (M : Algebra α)
    (s x y transition : RegionName) (length batch dim block : Nat) (u : State α) :
    ∃ v, FP.Structural.exec M (Kernels.complexForwardTyped s x y transition length batch dim block) u = some v := by
  obtain ⟨a, ha, hi⟩ : ∃ a,
      run M ((Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.take 7) u = some a ∧
      complexForwardInvariant block a := by
    simp [Kernels.ReadableSource.complex_forward_preserved, TypedSource.complexForward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexForwardInvariant]
  have hs : ∀ i a, i < length → complexForwardInvariant block a →
      ∃ b, run M (bodyAt (Kernels.complexForwardTyped s x y transition length batch dim block) 7)
        (a.setReg "t" .nat [] (fun _ => i)) = some b ∧ complexForwardInvariant block b := by
    intro i a _ hi
    rcases hi with ⟨v0, v1, v2, v3, v4, v5, h0, h1, h2, h3, h4, h5⟩
    simp [bodyAt, Kernels.ReadableSource.complex_forward_preserved, TypedSource.complexForward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexForwardInvariant,
      h0, h1, h2, h3, h4, h5]
  obtain ⟨b, hb, hi⟩ := loop_invariant M (P := fun _ => complexForwardInvariant block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.drop 8) b = some v := by
    rcases hi with ⟨v0, v1, v2, v3, v4, v5, h0, h1, h2, h3, h4, h5⟩
    simp [Kernels.ReadableSource.complex_forward_preserved, TypedSource.complexForward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      h0, h1, h2, h3, h4, h5]
  refine ⟨v, ?_⟩
  have source : (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody =
      (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.take 7 ++
        [.forLoop "t" length (bodyAt (Kernels.complexForwardTyped s x y transition length batch dim block) 7)] ++
          (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody.drop 8 := rfl
  change run M (Kernels.complexForwardTyped s x y transition length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, step, Option.bind_some, hb] using hv

def complexBackwardInvariant {α : Type} (block : Nat) (u : State α) : Prop :=
  ∃ v0 v1 v2 v3 v4 v5 v6 v7, u.regs .nat [block] "col_offsets" = some v0 ∧
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
      run M ((Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 9) u = some a ∧
      complexBackwardInvariant block a := by
    simp [Kernels.ReadableSource.complex_backward_preserved, TypedSource.complexBackward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexBackwardInvariant]
  have hs : ∀ i a, i < length → complexBackwardInvariant block a →
      ∃ b, run M (bodyAt (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) 9)
        (a.setReg "i" .nat [] (fun _ => i)) = some b ∧ complexBackwardInvariant block b := by
    intro i a _ hi
    rcases hi with ⟨v0, v1, v2, v3, v4, v5, v6, v7, h0, h1, h2, h3, h4, h5, h6, h7⟩
    by_cases hbranch : i < length - 1 <;>
      simp [hbranch, bodyAt, Kernels.ReadableSource.complex_backward_preserved, TypedSource.complexBackward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType, complexBackwardInvariant,
      h0, h1, h2, h3, h4, h5, h6, h7]
  obtain ⟨b, hb, hi⟩ := loop_invariant M (P := fun _ => complexBackwardInvariant block) hs
    0 (Nat.zero_le length) a hi
  obtain ⟨v, hv⟩ : ∃ v, run M ((Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 10) b = some v := by
    rcases hi with ⟨v0, v1, v2, v3, v4, v5, v6, v7, h0, h1, h2, h3, h4, h5, h6, h7⟩
    simp [Kernels.ReadableSource.complex_backward_preserved, TypedSource.complexBackward, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, numericLt, natLt, bop, store, Region.cast, ComputeDType.eraseDType,
      h0, h1, h2, h3, h4, h5, h6, h7]
  refine ⟨v, ?_⟩
  have source : (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody =
      (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.take 9 ++
        [.forLoop "i" length (bodyAt (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block) 9)] ++
          (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody.drop 10 := rfl
  change run M (Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block).surfaceBody u = some v
  rw [source, run_append, run_append, ha]
  simpa only [run, step, Option.bind_some, hb] using hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.FPExecution
