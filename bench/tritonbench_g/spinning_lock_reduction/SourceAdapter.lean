import VeriTile.Triton.DSL.ControlSource
import bench.tritonbench_g.spinning_lock_reduction.SourceReader

/-! Checked lowering from the shared concrete source tree to this task's
typed SC interpreter. Unsupported syntax is an error. The decoding depth is
a compiler resource limit, not a bound on source execution or spin retries. -/
namespace VeriTile.Bench.Repairs.SpinningLock.SourceAdapter
open Lean SourceReader
namespace CS
abbrev Expr := VeriTile.Triton.ControlSource.Expr
abbrev Argument := VeriTile.Triton.ControlSource.Argument
abbrev Statement := VeriTile.Triton.ControlSource.Statement
abbrev Program := VeriTile.Triton.ControlSource.Program
end CS

private def identifier : Name → String
  | .anonymous => ""
  | .str .anonymous part => part
  | .str parentName part => identifier parentName ++ "." ++ part
  | .num parentName index => identifier parentName ++ "." ++ Nat.repr index

private def name (n : TSyntax `ident) : String := identifier n.getId

private def decimal : List Char → Nat → Except String Nat
  | [], value => .ok value
  | digit :: rest, value =>
    if 48 ≤ digit.toNat && digit.toNat ≤ 57 then
      decimal rest (10 * value + digit.toNat - 48)
    else .error "the selected source dialect requires decimal natural literals"

private def numeral (raw : Syntax) : Except String Nat :=
  match raw.isLit? numLitKind with
  | none => .error "malformed natural literal"
  | some token =>
    match token.toList with
    | [] => .error "empty natural literal"
    | digits => decimal digits 0

private def param (n : TSyntax `ident) : Except String Param :=
  match name n with
  | "num_sms" => .ok .numSms
  | "k" => .ok .groupK
  | "M" => .ok .m
  | "N" => .ok .n
  | "stride_cm" => .ok .strideM
  | "stride_cn" => .ok .strideN
  | "BM" => .ok .bm
  | "BN" => .ok .bn
  | "PHASE" => .ok .phase
  | other => .error ("unsupported source parameter: " ++ other)

private def integer : Nat → Syntax → Except String IntExpr
  | 0, _ => .error "integer expression decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => integer fuel x.raw
    | `(tritonExpr| $n:num) => return .literal (← numeral n.raw)
    | `(tritonExpr| tl.program_id(0)) => .ok .pid
    | `(tritonExpr| tl.cast($x:tritonExpr, tl.int64)) => return .signed (← integer fuel x.raw)
    | `(tritonExpr| tl.cast($x:tritonExpr, tl.uint64)) => return .unsigned (← integer fuel x.raw)
    | `(tritonExpr| tl.arange(0, $size:ident)) =>
      match name size with
      | "BM" => .ok .row
      | "BN" => .ok .col
      | _ => .error "only the BM row and BN column ranges are supported"
    | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return .add (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => return .mul (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => return .div (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => return .rem (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $x:ident[:, None]) =>
      if name x ∈ ["rm1", "rm"] then .ok (.variable (name x))
      else .error "row projection requires a row source register"
    | `(tritonExpr| $x:ident[None, :]) =>
      if name x ∈ ["rn1", "rn"] then .ok (.variable (name x))
      else .error "column projection requires a column source register"
    | `(tritonExpr| $x:ident) =>
      if name x ∈ ["num_sms", "k", "M", "N", "stride_cm", "stride_cn", "BM", "BN", "PHASE"] then
        return .param (← param x)
      else .ok (.variable (name x))
    | _ => .error "unsupported spinning integer expression"

private def boolean : Nat → Syntax → Except String BoolExpr
  | 0, _ => .error "Boolean expression decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => boolean fuel x.raw
    | `(tritonExpr| ($x:ident < $y:tritonExpr)[:, None]) =>
      if name x != "rm" then .error "expected row mask projection" else
        return .lt (.variable (name x)) (← integer fuel y.raw)
    | `(tritonExpr| ($x:ident < $y:tritonExpr)[None, :]) =>
      if name x != "rn" then .error "expected column mask projection" else
        return .lt (.variable (name x)) (← integer fuel y.raw)
    | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => return .eq (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => return .ne (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => return .lt (← integer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $a:tritonExpr and $b:tritonExpr)
    | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => return .and (← boolean fuel a.raw) (← boolean fuel b.raw)
    | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) => return .or (← boolean fuel a.raw) (← boolean fuel b.raw)
    | `(tritonExpr| not $a:tritonExpr) => return .not (← boolean fuel a.raw)
    | `(tritonExpr| $x:ident) => .ok (.variable (name x))
    | _ => .error "unsupported spinning Boolean expression"

private def pointer : Nat → Syntax → Except String Pointer
  | 0, _ => .error "pointer expression decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => pointer fuel x.raw
    | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return .offset (← pointer fuel a.raw) (← integer fuel b.raw)
    | `(tritonExpr| $x:ident) =>
      match name x with
      | "P" => .ok (.base .partials)
      | "C" => .ok (.base .output)
      | "locks" => .ok (.base .locks)
      | other => .ok (.variable other)
    | _ => .error "unsupported spinning pointer expression"

private def condition (fuel : Nat) (raw : Syntax) : Except String Condition := do
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.atomic_cas($p:tritonExpr, $c:tritonExpr, $r:tritonExpr) != $v:tritonExpr) =>
    return .casNe (← pointer fuel p.raw) (← integer fuel c.raw)
      (← integer fuel r.raw) (← integer fuel v.raw)
  | _ => return .pure (← boolean fuel raw)

private def register (e : CS.Expr) : Except String String :=
  match (⟨e.toSyntax⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $x:ident) => .ok (name x)
  | _ => .error "expected source register"

private def natural (e : CS.Expr) : Except String Nat :=
  match (⟨e.toSyntax⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $x:num) => numeral x.raw
  | _ => .error "expected constant natural loop bound"

private def assignment (fuel : Nat) (target : String) (e : CS.Expr) : Except String Statement := do
  match (⟨e.toSyntax⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.zeros([$rows:ident, $cols:ident], dtype=tl.float32)) =>
    return .zeroFP32 target (← param rows) (← param cols)
  | `(tritonExpr| tl.load($p:tritonExpr)) => return .loadFP32 target (← pointer fuel p.raw)
  | _ =>
    if target ∈ ["P_", "C_"] then return .pointer target (← pointer fuel e.toSyntax)
    else if target ∈ ["is_reducer", "mask"] then return .boolean target (← boolean fuel e.toSyntax)
    else return .integer target (← integer fuel e.toSyntax)

private def positional (args : List CS.Argument) : Except String (List CS.Expr) :=
  args.mapM fun arg => if arg.keyword.isNone then .ok arg.value else .error "unexpected source keyword"

private def effect (fuel : Nat) (function : String) (args : List CS.Argument) : Except String Statement := do
  match function, args with
  | "tl.store", [p,v] =>
    let values ← positional [p,v]
    match values with
    | [p,v] => return .storeFP32 (← pointer fuel p.toSyntax) (← register v) none
    | _ => .error "invalid store arguments"
  | "tl.store", [p,v,m] =>
    if p.keyword.isSome || v.keyword.isSome || m.keyword != some "mask" then
      .error "invalid store keyword arguments"
    else
      let p ← pointer fuel p.value.toSyntax
      let v ← register v.value
      let m ← boolean fuel m.value.toSyntax
      return .storeFP32 p v (some m)
  | _, _ =>
    let values ← positional args
    match function, values with
    | "tl.static_assert", [e] => return .staticAssert (← boolean fuel e.toSyntax)
    | "tl.debug_barrier", [] => .ok .blockBarrier
    | "tl.atomic_xchg", [p,v] => return .xchg (← pointer fuel p.toSyntax) (← integer fuel v.toSyntax)
    | _, _ => .error ("unsupported spinning source call: " ++ function)

private def statement : Nat → CS.Statement → Except String Statement
  | 0, _ => .error "statement decoding depth exceeded"
  | fuel + 1, source => do
    match source with
    | .assign target value => assignment fuel target value
    | .update target "+" value =>
      if target == "acc" then return .addFP32 target (← register value)
      else return .integer target (.add (.variable target) (← integer fuel value.toSyntax))
    | .call function args => effect fuel function args
    | .branch test yes no =>
      return .branch (← boolean fuel test.toSyntax)
        (← yes.mapM (statement fuel)) (← no.mapM (statement fuel))
    | .range index [lo,hi] body =>
      return .forRange index (← natural lo) (← natural hi) (← body.mapM (statement fuel))
    | .whileLoop test body =>
      return .whileLoop (← condition fuel test.toSyntax) (← body.mapM (statement fuel))
    | .returnValues [] => .ok .returnNow
    | .pass => .ok .pass
    | _ => .error "unsupported spinning source statement"

/-- Every selected source statement is decoded, including nested loops and
all CAS operands. The complete program must decode successfully before use. -/
def decode (source : CS.Program) : Except String (List Statement) :=
  source.mapM (statement 64)

end VeriTile.Bench.Repairs.SpinningLock.SourceAdapter
