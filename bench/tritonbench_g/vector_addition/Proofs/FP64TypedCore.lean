import bench.tritonbench_g.vector_addition.Proofs.Binary64Words

/-! A task-local typed scalar source and interpreter. Every storage dtype has
its own indexed word; fp64 is genuinely BitVec 64, with no Real carrier.
This module proves a per-lane structural route theorem under explicit numerical
callbacks. It does not prove grid/memory/address/compiler/native refinement. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

inductive Width where
  | w8 | w16 | w32 | w64
  deriving DecidableEq, Repr

def Width.bits : Width → Nat
  | .w8 => 8 | .w16 => 16 | .w32 => 32 | .w64 => 64

inductive Storage where
  | boolean | signed (width : Width) | unsigned (width : Width)
  | fp16 | bf16 | fp32 | fp64 | f8e4fn | f8e5
  deriving DecidableEq, Repr

def Storage.width : Storage → Nat
  | .boolean => 8
  | .signed w | .unsigned w => w.bits
  | .fp16 | .bf16 => 16
  | .fp32 => 32 | .fp64 => 64
  | .f8e4fn | .f8e5 => 8

inductive RegType where
  | storage (dtype : Storage)
  | predicate
  deriving DecidableEq, Repr

def RegType.width : RegType → Nat
  | .storage d => d.width | .predicate => 1

/-- The index separates equal-width formats as well as unequal-width ones. -/
structure Word (d : RegType) where
  bits : BitVec d.width
  deriving DecidableEq, Repr

abbrev F64 := RegType.storage .fp64
abbrev F32 := RegType.storage .fp32

def decode64 (x : Word F64) := Binary64Words.decode x.bits
def encode64 (f : Binary64Words.Fields) : Word F64 := ⟨Binary64Words.encode f⟩

theorem word64_roundtrip (x : Word F64) :
    encode64 (Binary64Words.fields x.bits) = x := by
  cases x
  simp [encode64, Binary64Words.encode_fields]

/-- Numerical operations are explicit parameters. Floating conversions/addition
are total word functions, including specials. Only float-to-integer operations
may be undefined. No numerical or native-conformance law is assumed silently. -/
structure Numerics where
  signedTo64 : (w : Width) → Word (.storage (.signed w)) → Word F64
  unsignedTo64 : (w : Width) → Word (.storage (.unsigned w)) → Word F64
  halfTo32 : Word (.storage .fp16) → Word F32
  bfloatTo32 : Word (.storage .bf16) → Word F32
  singleTo64 : Word F32 → Word F64
  e4To64 : Word (.storage .f8e4fn) → Word F64
  e5To64 : Word (.storage .f8e5) → Word F64
  toSigned : (w : Width) → Word F64 → Option (Word (.storage (.signed w)))
  toUnsigned : (w : Width) → Word F64 → Option (Word (.storage (.unsigned w)))
  toHalf : Word F64 → Word (.storage .fp16)
  toBFloat : Word F64 → Word (.storage .bf16)
  toSingle : Word F64 → Word F32
  toE4RTNE : Word F64 → Word (.storage .f8e4fn)
  toE5RTNE : Word F64 → Word (.storage .f8e5)
  add64 : Word F64 → Word F64 → Word F64

inductive Cast : RegType → RegType → Type where
  | boolLoad : Cast (.storage .boolean) .predicate
  | boolTo64 : Cast .predicate F64
  | boolStore : Cast (.storage (.signed .w8)) (.storage .boolean)
  | signedTo64 (w : Width) : Cast (.storage (.signed w)) F64
  | unsignedTo64 (w : Width) : Cast (.storage (.unsigned w)) F64
  | halfTo32 : Cast (.storage .fp16) F32
  | bfloatTo32 : Cast (.storage .bf16) F32
  | singleTo64 : Cast F32 F64
  | e4To64 : Cast (.storage .f8e4fn) F64
  | e5To64 : Cast (.storage .f8e5) F64
  | toSigned (w : Width) : Cast F64 (.storage (.signed w))
  | toUnsigned (w : Width) : Cast F64 (.storage (.unsigned w))
  | toHalf : Cast F64 (.storage .fp16)
  | toBFloat : Cast F64 (.storage .bf16)
  | toSingle : Cast F64 F32
  | toE4RTNE : Cast F64 (.storage .f8e4fn)
  | toE5RTNE : Cast F64 (.storage .f8e5)

/-- Bool input uses byte != 0 then unsigned i1→f64. Bool output preserves the
entire fptosi i8 byte, including non-0/1 bytes; it is not normalized on store. -/
def evalCast (b : Numerics) : Cast s t → Word s → Option (Word t)
  | .boolLoad, v => some ⟨BitVec.ofBool (v.bits != 0)⟩
  | .boolTo64, v => some ⟨if v.bits.msb then 0x3ff0000000000000#64 else 0#64⟩
  | .boolStore, v => some ⟨v.bits⟩
  | .signedTo64 w, v => some (b.signedTo64 w v)
  | .unsignedTo64 w, v => some (b.unsignedTo64 w v)
  | .halfTo32, v => some (b.halfTo32 v)
  | .bfloatTo32, v => some (b.bfloatTo32 v)
  | .singleTo64, v => some (b.singleTo64 v)
  | .e4To64, v => some (b.e4To64 v)
  | .e5To64, v => some (b.e5To64 v)
  | .toSigned w, v => b.toSigned w v
  | .toUnsigned w, v => b.toUnsigned w v
  | .toHalf, v => some (b.toHalf v)
  | .toBFloat, v => some (b.toBFloat v)
  | .toSingle, v => some (b.toSingle v)
  | .toE4RTNE, v => some (b.toE4RTNE v)
  | .toE5RTNE, v => some (b.toE5RTNE v)

inductive Expr (xD yD : Storage) : RegType → Type where
  | x : Expr xD yD (.storage xD)
  | y : Expr xD yD (.storage yD)
  | cast : Cast s t → Expr xD yD s → Expr xD yD t
  | add64 : Expr xD yD F64 → Expr xD yD F64 → Expr xD yD F64

def Expr.eval (b : Numerics) (x : Word (.storage xD)) (y : Word (.storage yD)) :
    Expr xD yD d → Option (Word d)
  | .x => some x
  | .y => some y
  | .cast c e => (e.eval b x y).bind (evalCast b c)
  | .add64 l r => do
    let a ← l.eval b x y
    let z ← r.eval b x y
    pure (b.add64 a z)

/-- Exactly the source promotion route, including the explicit fp32 stage for
fp16/bf16 inputs and direct fp8→fp64 operations. -/
def promote : (d : Storage) → Expr xD yD (.storage d) → Expr xD yD F64
  | .boolean, e => .cast .boolTo64 (.cast .boolLoad e)
  | .signed w, e => .cast (.signedTo64 w) e
  | .unsigned w, e => .cast (.unsignedTo64 w) e
  | .fp16, e => .cast .singleTo64 (.cast .halfTo32 e)
  | .bf16, e => .cast .singleTo64 (.cast .bfloatTo32 e)
  | .fp32, e => .cast .singleTo64 e
  | .fp64, e => e
  | .f8e4fn, e => .cast .e4To64 e
  | .f8e5, e => .cast .e5To64 e

/-- Direct f64→output conversion; no invented intermediate fp32 narrowing.
The destination dtype is always the original x storage dtype. -/
def stored : (d : Storage) → Expr xD yD F64 → Expr xD yD (.storage d)
  | .boolean, e => .cast .boolStore (.cast (.toSigned .w8) e)
  | .signed w, e => .cast (.toSigned w) e
  | .unsigned w, e => .cast (.toUnsigned w) e
  | .fp16, e => .cast .toHalf e
  | .bf16, e => .cast .toBFloat e
  | .fp32, e => .cast .toSingle e
  | .fp64, e => e
  | .f8e4fn, e => .cast .toE4RTNE e
  | .f8e5, e => .cast .toE5RTNE e

/-- Typed scalar expression for a supported f64-compute profile. Applicability
is stated separately; constructing other pairs is not a source support claim. -/
def source (xD yD : Storage) : Expr xD yD (.storage xD) :=
  stored xD (.add64 (promote xD .x) (promote yD .y))

def Supported (xD yD : Storage) : Prop := xD = .fp64 ∨ yD = .fp64

/-- Independent word-level formula. This does not call Expr.eval/source. -/
def input64 (b : Numerics) : (d : Storage) → Word (.storage d) → Word F64
  | .boolean, v => ⟨if v.bits != 0 then 0x3ff0000000000000#64 else 0#64⟩
  | .signed w, v => b.signedTo64 w v
  | .unsigned w, v => b.unsignedTo64 w v
  | .fp16, v => b.singleTo64 (b.halfTo32 v)
  | .bf16, v => b.singleTo64 (b.bfloatTo32 v)
  | .fp32, v => b.singleTo64 v
  | .fp64, v => v
  | .f8e4fn, v => b.e4To64 v
  | .f8e5, v => b.e5To64 v

def output64 (b : Numerics) : (d : Storage) → Word F64 → Option (Word (.storage d))
  | .boolean, v => (b.toSigned .w8 v).map (fun w => ⟨w.bits⟩)
  | .signed w, v => b.toSigned w v
  | .unsigned w, v => b.toUnsigned w v
  | .fp16, v => some (b.toHalf v)
  | .bf16, v => some (b.toBFloat v)
  | .fp32, v => some (b.toSingle v)
  | .fp64, v => some v
  | .f8e4fn, v => some (b.toE4RTNE v)
  | .f8e5, v => some (b.toE5RTNE v)

def result (b : Numerics) (xD yD : Storage)
    (x : Word (.storage xD)) (y : Word (.storage yD)) : Option (Word (.storage xD)) :=
  output64 b xD (b.add64 (input64 b xD x) (input64 b yD y))

theorem eval_promote (b : Numerics) (x : Word (.storage xD)) (y : Word (.storage yD))
    (d : Storage) (e : Expr xD yD (.storage d)) :
    (promote d e).eval b x y = (e.eval b x y).map (input64 b d) := by
  cases d <;> cases h : e.eval b x y <;>
    simp [promote, Expr.eval, evalCast, input64, h]
  case boolean.some v =>
    change (if (BitVec.ofBool (v.bits != 0)).msb then 0x3ff0000000000000#64 else 0#64) = _
    rw [BitVec.msb_ofBool]
    by_cases hz : v.bits = 0 <;> simp [hz] <;> rfl

theorem eval_stored (b : Numerics) (x : Word (.storage xD)) (y : Word (.storage yD))
    (d : Storage) (e : Expr xD yD F64) :
    (stored d e).eval b x y = (e.eval b x y).bind (output64 b d) := by
  cases d <;> cases h : e.eval b x y <;>
    simp [stored, Expr.eval, evalCast, output64, h, Option.map_eq_bind]
  case boolean.some v =>
    cases b.toSigned .w8 v <;> rfl

theorem eval_source (b : Numerics) (xD yD : Storage)
    (x : Word (.storage xD)) (y : Word (.storage yD)) :
    (source xD yD).eval b x y = result b xD yD x y := by
  rw [source, eval_stored]
  simp [Expr.eval, eval_promote, result]

inductive Outcome (d : Storage) where
  | masked
  | loadUnavailable
  | conversionUndefined
  | write (value : Word (.storage d))
  deriving DecidableEq, Repr

/-- Per-lane write request only. The externally supplied mask is not a proved
address calculation. Missing active loads and undefined integer conversions
remain explicit. Masked lanes need no invented `other` value. -/
def runLane (b : Numerics) (xD yD : Storage) (active : Bool)
    (x : Option (Word (.storage xD))) (y : Option (Word (.storage yD))) : Outcome xD :=
  if active then
    match x, y with
    | some a, some z =>
      match (source xD yD).eval b a z with
      | some v => .write v
      | none => .conversionUndefined
    | _, _ => .loadUnavailable
  else .masked

theorem masked_lane (b : Numerics) (xD yD : Storage)
    (x : Option (Word (.storage xD))) (y : Option (Word (.storage yD))) :
    runLane b xD yD false x y = .masked := rfl

theorem active_lane (b : Numerics) (xD yD : Storage)
    (x : Word (.storage xD)) (y : Word (.storage yD)) :
    runLane b xD yD true (some x) (some y) =
      match result b xD yD x y with
      | some v => .write v
      | none => .conversionUndefined := by
  simp [runLane, eval_source]

theorem missing_x (b : Numerics) (xD yD : Storage)
    (y : Option (Word (.storage yD))) :
    runLane b xD yD true none y = .loadUnavailable := by
  cases y <;> rfl

theorem missing_y (b : Numerics) (xD yD : Storage)
    (x : Option (Word (.storage xD))) :
    runLane b xD yD true x none = .loadUnavailable := by
  cases x <;> rfl

def Storage.isFloating : Storage → Bool
  | .fp16 | .bf16 | .fp32 | .fp64 | .f8e4fn | .f8e5 => true
  | _ => false

/-- Arbitrary bit patterns, including specials, have a write request for every
floating output under the chosen total word callbacks. No native claim follows. -/
theorem floating_output_total (b : Numerics) (xD yD : Storage)
    (hf : xD.isFloating = true)
    (x : Word (.storage xD)) (y : Word (.storage yD)) :
    ∃ v, runLane b xD yD true (some x) (some y) = .write v := by
  rw [active_lane]
  cases xD <;> simp [Storage.isFloating] at hf
  all_goals simp [result, output64]

theorem boolean_store_preserves_byte (b : Numerics) (v : BitVec 8) :
    evalCast b .boolStore ⟨v⟩ = some ⟨v⟩ := rfl

theorem boolean_load_nonzero (b : Numerics) (v : BitVec 8) :
    evalCast b .boolLoad ⟨v⟩ = some ⟨BitVec.ofBool (v != 0)⟩ := rfl

#specNonCircular result avoiding [source, Expr.eval, runLane]
#axiomsClean word64_roundtrip
#axiomsClean eval_promote
#axiomsClean eval_stored
#axiomsClean eval_source
#axiomsClean masked_lane
#axiomsClean active_lane
#axiomsClean missing_x
#axiomsClean missing_y
#axiomsClean floating_output_total
#axiomsClean boolean_store_preserves_byte
#axiomsClean boolean_load_nonzero
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore
