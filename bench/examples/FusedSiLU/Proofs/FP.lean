import bench.examples.FusedSiLU.IO
import VeriTile.Triton.Float.SoftwarePipelineIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv
open VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

def siluValue {α : Type} (M : Algebra α) (z : α) : α :=
  M.binary (some .fp32) .real .mul z (M.unary (some .fp32) .sigmoid z)

def resultValue {α : Type} (M : Algebra α) (x g r : α) : α :=
  M.cast none .real .bf16 (M.binary (some .fp32) .real .add r
    (siluValue M (M.binary (some .fp32) .real .mul x g)))

theorem gate_run {α : Type} [Inhabited α] (M : Algebra α) (hload : ∀ v, M.fp32Load v = v) (B : Nat) (s : State α)
    (xs gs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    : ∃ t, FP.Structural.exec M (siluStepGate "x" "gate" "z" B) s = some t ∧
      (∀ i : Fin B, t.mem "z" (s.pids 0 * B + i.val) = Cell.mk .real (M.binary (some .fp32) .real .mul (xs i) (gs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [hload, ComputeDType.eraseDType, siluStepGate, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, bop, store]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hload, ComputeDType.eraseDType, hxs, hgs]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem silu_run {α : Type} [Inhabited α] (M : Algebra α) (hload : ∀ v, M.fp32Load v = v) (B : Nat) (s : State α)
    (zs : Fin B → α)
    (hzs : ∀ i : Fin B, (s.mem "z" (s.pids 0 * B + i.val)).read .real = zs i)
    : ∃ t, FP.Structural.exec M (siluStepSilu "z" "silu" B) s = some t ∧
      (∀ i : Fin B, t.mem "silu" (s.pids 0 * B + i.val) = Cell.mk .real (siluValue M (zs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [hload, ComputeDType.eraseDType, siluStepSilu, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, bop, store,
    Function.comp_def]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hload, ComputeDType.eraseDType, hzs, siluValue]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem residual_run {α : Type} [Inhabited α] (M : Algebra α) (hload : ∀ v, M.fp32Load v = v) (B : Nat) (s : State α)
    (ys rs : Fin B → α)
    (hys : ∀ i : Fin B, (s.mem "silu" (s.pids 0 * B + i.val)).read .real = ys i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i)
    : ∃ t, FP.Structural.exec M (siluStepResidual "silu" "residual" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (M.cast none .real .bf16 (M.binary (some .fp32) .real .add (rs i) (ys i)))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [hload, ComputeDType.eraseDType, siluStepResidual, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, bop, store, ofFloat, toFloat]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hload, ComputeDType.eraseDType, hys, hrs]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem fused_run {α : Type} [Inhabited α] (M : Algebra α) (hload : ∀ v, M.fp32Load v = v) (B : Nat) (s : State α)
    (xs gs rs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i)
    : ∃ t, FP.Structural.exec M (fusedSiLUKernel "x" "gate" "residual" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (resultValue M (xs i) (gs i) (rs i))) ∧
      t.pids = s.pids ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [hload, ComputeDType.eraseDType, fusedSiLUKernel, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, bop, store, ofFloat, toFloat,
    Function.comp_def]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
    simp [hload, ComputeDType.eraseDType, hxs, hgs, hrs, siluValue, resultValue]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

theorem unfused_run {α : Type} [Inhabited α] (M : Algebra α) (hload : ∀ v, M.fp32Load v = v) (B : Nat) (s : State α)
    (xs gs rs : Fin B → α)
    (hxs : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i)
    (hgs : ∀ i : Fin B, (s.mem "gate" (s.pids 0 * B + i.val)).read .real = gs i)
    (hrs : ∀ i : Fin B, (s.mem "residual" (s.pids 0 * B + i.val)).read .real = rs i) :
    ∃ t, FP.Structural.exec M (unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B) s = some t ∧
      (∀ i : Fin B, t.mem "out" (s.pids 0 * B + i.val) = Cell.mk .bf16 (resultValue M (xs i) (gs i) (rs i))) ∧
      (∀ (r : RegionName) o,
        (r ≠ "out" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        (r ≠ "z" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        (r ≠ "silu" ∨ ∀ i : Fin B, o ≠ s.pids 0 * B + i.val) →
        t.mem r o = s.mem r o) := by
  obtain ⟨s1, he1, hz, hp1, hf1⟩ := gate_run M hload B s xs gs hxs hgs
  let zs : Fin B → α := fun i => M.binary (some .fp32) .real .mul (xs i) (gs i)
  obtain ⟨s2, he2, hy, hp2, hf2⟩ := silu_run M hload B s1 zs (by
    intro i
    rw [hp1, hz i]
    rfl)
  rw [hp1] at hy hp2 hf2
  obtain ⟨s3, he3, hv, hp3, hf3⟩ := residual_run M hload B s2 (fun i => siluValue M (zs i)) rs
    (by intro i; rw [hp2, hy i]; rfl)
    (by
      intro i
      rw [hp2, hf2 "residual" _ (Or.inl (by decide)), hf1 "residual" _ (Or.inl (by decide))]
      exact hrs i)
  rw [hp2] at hv hp3 hf3
  refine ⟨s3, ?_, ?_, ?_⟩
  · simp only [unfusedSiLUKernel, exec_seq_cons, he1, Option.bind_some, he2, he3, exec_seq_nil]
  · simpa only [resultValue, zs] using hv
  · intro r o ho hz hs
    exact (hf3 r o ho).trans ((hf2 r o hs).trans (hf1 r o hz))

open scoped VeriTile.Spec

/- Internal generic proof assumes identity FP32 reloads. The public software
word interpretation discharges this by definition, rather than assuming a
numerical identity about an arbitrary algebra. -/

def WithIdentityLoadsEquiv (lhs rhs : KernelIO₃) : Prop :=
  PrivateScratch lhs ∧ PrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α), (∀ v, M.fp32Load v = v) → ∀ (s : State α),
    ∃ a b, exec M lhs.kernel s = some a ∧ exec M rhs.kernel s = some b ∧
      (∀ i : Fin lhs.Bout, a.mem lhs.out (lhs.write (s.pids 0) + i.val) = b.mem rhs.out (rhs.write (s.pids 0) + i.val)) ∧
      Frame lhs s a ∧ Frame rhs s b

namespace Execution

abbrev unfusedIO := FusedSiLU.IO.unfusedIO
abbrev fusedIO := FusedSiLU.IO.fusedIO

open scoped VeriTile.Spec

/-- Compose execution, numerical laws and the memory frames. -/
theorem equivalent (B : Nat) :
    WithIdentityLoadsEquiv (fusedIO B) (unfusedIO B) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [PrivateScratch, fusedIO, FusedSiLU.IO.fusedIO]
  · intro p hp
    simp only [unfusedIO, FusedSiLU.IO.unfusedIO, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> simp only [unfusedIO, FusedSiLU.IO.unfusedIO] <;> decide
  · intro α _ M hload s
    let xs : Fin B → α := fun i => (s.mem "x" (s.pids 0 * B + i.val)).read .real
    let gs : Fin B → α := fun i => (s.mem "gate" (s.pids 0 * B + i.val)).read .real
    let rs : Fin B → α := fun i => (s.mem "residual" (s.pids 0 * B + i.val)).read .real
    obtain ⟨a, ha, hva, _, hfa⟩ := fused_run M hload B s xs gs rs (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := unfused_run M hload B s xs gs rs (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
    refine ⟨a, b, ha, hb, fun i => (hva i).trans (hvb i).symm, ?_, ?_⟩
    · intro r o ho _; exact hfa r o ho
    · intro r o ho hs
      apply hfb r o ho
      · by_cases hr : r = "z"
        · exact Or.inr (hs { buf := "z", win := fun pid => pid * B, len := B } (by simp [unfusedIO, FusedSiLU.IO.unfusedIO]) hr)
        · exact Or.inl hr
      · by_cases hr : r = "silu"
        · exact Or.inr (hs { buf := "silu", win := fun pid => pid * B, len := B } (by simp [unfusedIO, FusedSiLU.IO.unfusedIO]) hr)
        · exact Or.inl hr

end Execution

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
