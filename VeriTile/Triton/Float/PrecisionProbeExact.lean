import VeriTile.Triton.Float.PrecisionIdentityLaws

/-! Typed certificates for the eight precision-only historical instances.
The two BF16-input profiles share these particular source graphs: the probes
select their casts independently of the arithmetic-compute metadata. Every
certificate retains the final typed store and proves equality of full states.
-/
namespace VeriTile.Triton.FP.PrecisionProbeExact
open Structural PrecisionIdentityLaws

inductive Instance where
  | roundIdemFp32
  | accWidenFp32
  | roundIdemBf16 (widenReturn mixed : Bool)
  | castMoveBf16 (mixed : Bool)
  deriving DecidableEq, Repr

def Instance.ruleID : Instance → String
  | .roundIdemFp32 | .roundIdemBf16 false _ => "ROUND-IDEM"
  | .roundIdemBf16 true _ => "BF16-WIDEN-RETURN"
  | .castMoveBf16 _ => "CAST-MOVE"
  | .accWidenFp32 => "ACC-WIDEN"

def Instance.profileName : Instance → String
  | .roundIdemFp32 | .accWidenFp32 => "fp32"
  | .roundIdemBf16 _ false | .castMoveBf16 false => "bf16"
  | .roundIdemBf16 _ true | .castMoveBf16 true => "bf16_fp32"

def Instance.operands : Instance → List String
  | .roundIdemFp32 | .roundIdemBf16 _ _ => ["a"]
  | .castMoveBf16 _ => ["a", "b"]
  | .accWidenFp32 => ["a", "b", "c"]

def all : List Instance := [.roundIdemFp32, .accWidenFp32,
  .roundIdemBf16 false false, .roundIdemBf16 false true,
  .roundIdemBf16 true false, .roundIdemBf16 true true,
  .castMoveBf16 false, .castMoveBf16 true]

def RoundIdemFp32Sound : Prop :=
  ∀ {α : Type} [Inhabited α] (M : Algebra α) (out : Region .fp32)
    (offset : Nat) (s : State α) (a : α),
    s.regs .fp32 [] "a" = some (fun _ => a) →
    run (sourceCopies M) (scalarStore out offset roundIdemFp32Lhs) s =
      run (sourceCopies M) (scalarStore out offset roundIdemFp32Rhs) s

def AccWidenFp32Sound : Prop :=
  ∀ {α : Type} [Inhabited α] (M : Algebra α) (out : Region .fp32)
    (offset : Nat) (s : State α) (a b c : α),
    s.regs .fp32 [] "a" = some (fun _ => a) →
    s.regs .fp32 [] "b" = some (fun _ => b) →
    s.regs .fp32 [] "c" = some (fun _ => c) →
    run (sourceCopies M) (scalarStore out offset accWidenFp32Lhs) s =
      run (sourceCopies M) (scalarStore out offset accWidenFp32Rhs) s

def RoundIdemBf16Sound : Prop :=
  ∀ [Inhabited CastValue] (rest : Algebra CastValue) (out : Region .bf16)
    (offset : Nat) (s : State CastValue) (a : FP.Value .bf16),
    s.regs .bf16 [] "a" = some (fun _ => Sum.inl a) →
    a.toRat?.isSome = true →
    run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset roundIdemBf16Lhs) s =
      run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset roundIdemBf16Rhs) s

def CastMoveBf16Sound : Prop :=
  ∀ [Inhabited CastValue] (rest : Algebra CastValue) (out : Region .bf16)
    (offset : Nat) (s : State CastValue) (a b : FP.Value .bf16),
    s.regs .bf16 [] "a" = some (fun _ => Sum.inl a) →
    s.regs .bf16 [] "b" = some (fun _ => Sum.inl b) →
    a.toRat?.isSome = true → b.toRat?.isSome = true →
    run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset castMoveBf16Lhs) s =
      run (sourceCopies (bf16FP32Casts rest)) (scalarStore out offset castMoveBf16Rhs) s

/-- The obligation is fixed by the instance; clients cannot supply an
unrelated proposition or a declaration-name string as its certificate. -/
def Certificate : Instance → Prop
  | .roundIdemFp32 => RoundIdemFp32Sound
  | .accWidenFp32 => AccWidenFp32Sound
  | .roundIdemBf16 _ _ => RoundIdemBf16Sound
  | .castMoveBf16 _ => CastMoveBf16Sound

private theorem not_nan_of_finite {f : Format} (a : FP.Value f)
    (ha : a.toRat?.isSome = true) : a.isNaN = false := by
  cases hd : a.decode with
  | finite s m => exact CastExact.isNaN_of_decode_finite a s m hd
  | infinity s => simp [FP.Value.toRat?, hd] at ha
  | nan s q p => simp [FP.Value.toRat?, hd] at ha

theorem all_certificates (i : Instance) : Certificate i := by
  cases i with
  | roundIdemFp32 =>
    intro α inst M out offset s a ha
    have h := roundIdemFp32_store M out offset s a ha
    exact h.1.trans h.2.symm
  | accWidenFp32 =>
    intro α inst M out offset s a b c ha hb hc
    have h := accWidenFp32_store M out offset s a b c ha hb hc
    exact h.1.trans h.2.symm
  | roundIdemBf16 widened mixed =>
    intro inst rest out offset s a ha hfinite
    have h := roundIdemBf16_store_concreteCasts rest out offset s a ha
      (not_nan_of_finite a hfinite)
    exact h.1.trans h.2.symm
  | castMoveBf16 mixed =>
    intro inst rest out offset s a b ha hb hfiniteA hfiniteB
    have h := castMoveBf16_store_concreteCasts rest out offset s a b ha hb
      (not_nan_of_finite a hfiniteA) (not_nan_of_finite b hfiniteB)
    exact h.1.trans h.2.symm

#axiomsClean all_certificates

end VeriTile.Triton.FP.PrecisionProbeExact
