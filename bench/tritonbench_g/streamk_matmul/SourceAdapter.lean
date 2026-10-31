import VeriTile.Triton.DSL.ControlSource
import bench.tritonbench_g.streamk_matmul.SourceReader

/-! Total checked lowering of the complete shared source tree into the
existing typed StreamK reader. Source helpers retain all arguments; lowering
checks their binding convention rather than accepting opaque callbacks.
The depth limit bounds syntax decoding, not loop execution. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.SourceAdapter
open Lean Reader
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

private def expr : Nat → Syntax → Except String Reader.Expr
  | 0, _ => .error "integer expression decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => expr fuel x.raw
    | `(tritonExpr| end) => .ok (.variable "end")
    | `(tritonExpr| $n:num) => return .literal (← numeral n.raw)
    | `(tritonExpr| tl.program_id(0)) => .ok .pid
    | `(tritonExpr| tl.cast($x:tritonExpr, tl.int64))
    | `(tritonExpr| tl.full((), $x:tritonExpr, tl.int64)) => return .signed64 (← expr fuel x.raw)
    | `(tritonExpr| tl.arange(0, $n:ident)) =>
      match name n with
      | "BLOCK_M" => .ok .laneM
      | "BLOCK_N" => .ok .laneN
      | "BLOCK_K" => .ok .laneK
      | _ => .error "unsupported source block range"
    | `(tritonExpr| $x:tritonExpr[:, None])
    | `(tritonExpr| $x:tritonExpr[None, :]) => expr fuel x.raw
    | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return .add (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => return .sub (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => return .mul (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => return .div (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => return .rem (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => return .minimum (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => return .cdiv (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $x:ident) => .ok (.variable (name x))
    | _ => .error "unsupported StreamK integer expression"

private def test : Nat → Syntax → Except String Test
  | 0, _ => .error "condition decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => test fuel x.raw
    | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => return .eq (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => return .ne (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => return .lt (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => return .gt (← expr fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => return .and (← test fuel a.raw) (← test fuel b.raw)
    | `(tritonExpr| $x:ident) => .ok (.variable (name x))
    | _ => .error "unsupported StreamK condition"

private def pointer : Nat → Syntax → Except String Pointer
  | 0, _ => .error "pointer expression decoding depth exceeded"
  | fuel + 1, raw => do
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => pointer fuel x.raw
    | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return .offset (← pointer fuel a.raw) (← expr fuel b.raw)
    | `(tritonExpr| $x:ident) => .ok (.variable (name x))
    | _ => .error "unsupported StreamK pointer expression"

private def register (e : CS.Expr) : Except String String :=
  match (⟨e.toSyntax⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $x:ident) => .ok (name x)
  | _ => .error "expected source register"

private def positional (args : List CS.Argument) : Except String (List CS.Expr) :=
  args.mapM fun arg => if arg.keyword.isNone then .ok arg.value else .error "unexpected source keyword"
private def checkArgs : List CS.Expr → List String → Except String Unit
  | [], [] => .ok ()
  | arg :: args, expected :: names => do
    if (← register arg) != expected then .error "helper source binding changed"
    else checkArgs args names
  | _, _ => .error "wrong helper source arity"

private def assignment (fuel : Nat) (target : String) (raw : Syntax) : Except String Statement := do
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.zeros([$m:tritonExpr, $n:tritonExpr], dtype=tl.float32)) =>
    return .zeros target (← expr fuel m.raw) (← expr fuel n.raw)
  | `(tritonExpr| tl.cast($y:ident, tl.float16)) =>
    if target != name y then .error "unsupported conversion target" else .ok (.half target)
  | `(tritonExpr| tl.load($p:tritonExpr, mask=$m:tritonExpr, other=0.0)) =>
    return .load target (← pointer fuel p.raw) (some (← test fuel m.raw))
  | `(tritonExpr| tl.load($p:tritonExpr)) => return .load target (← pointer fuel p.raw) none
  | _ =>
    if target ∈ ["A","B","C","C_"] then return .pointer target (← pointer fuel raw)
    else if target == "mask" then return .boolean target (← test fuel raw)
    else return .integer target (← expr fuel raw)

private def tileCall (fuel : Nat) (targets : List String) (function : String)
    (args : List CS.Argument) : Except String Statement := do
  if targets != ["pid_m","pid_n"] then .error "unsupported mapping outputs" else
    let args ← positional args
    match args with
    | tile :: tail =>
      checkArgs tail ["M","N","K","BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M"]
      let swizzle ← match function with
        | "swizzle_tile" => .ok Bool.true
        | "linear_tile" => .ok Bool.false
        | _ => .error "unsupported mapping helper"
      return .tileCall swizzle (← expr fuel tile.toSyntax)
    | [] => .error "missing mapping arguments"

private def effect (fuel : Nat) (function : String) (args : List CS.Argument) : Except String Statement := do
  match function, args with
  | "tl.store", [p,v] =>
    let values ← positional [p,v]
    match values with
    | [p,v] => return .store (← pointer fuel p.toSyntax) (← register v) none
    | _ => .error "invalid store arguments"
  | "tl.store", [p,v,m] | "tl.atomic_add", [p,v,m] =>
    if p.keyword.isSome || v.keyword.isSome || m.keyword != some "mask" then
      .error "invalid memory-effect keyword arguments"
    else
      let p ← pointer fuel p.value.toSyntax
      let v ← register v.value
      let m ← test fuel m.value.toSyntax
      if function == "tl.store" then return .store p v (some m)
      else return .atomic p v m
  | "mac_loop", _ =>
    let args ← positional args
    checkArgs args ["A","B","C","M","N","K","partials","pid","stride_am","stride_ak",
      "stride_bk","stride_bn","stride_cm","stride_cn","iters_per_tile","start_iter","end_iter",
      "BLOCK_M","BLOCK_N","BLOCK_K","ACC_TYPE","GROUP_M"]
    .ok .macCall
  | _, _ => .error ("unsupported StreamK source call: " ++ function)

private def statement : Nat → CS.Statement → Except String Statement
  | 0, _ => .error "statement decoding depth exceeded"
  | fuel + 1, source => do
    match source with
    | .assign target value => assignment fuel target value.toSyntax
    | .update target "+" value =>
      match (⟨value.toSyntax⟩ : TSyntax `tritonExpr) with
      | `(tritonExpr| tl.dot($a:ident, $b:ident)) => .ok (.dotAdd target (name a) (name b))
      | _ =>
        if target ∈ ["A","B"] then
          return .pointer target (.offset (.variable target) (← expr fuel value.toSyntax))
        else .error "unsupported source inplace update"
    | .bindCall targets function args => tileCall fuel targets function args
    | .call function args => effect fuel function args
    | .branch condition yes no =>
      return .branch (← test fuel condition.toSyntax)
        (← yes.mapM (statement fuel)) (← no.mapM (statement fuel))
    | .range index [lo,hi] body =>
      return .range index (← expr fuel lo.toSyntax) (← expr fuel hi.toSyntax) (← body.mapM (statement fuel))
    | .whileLoop condition body => return .loop (← test fuel condition.toSyntax) (← body.mapM (statement fuel))
    | .returnValues [a,b] => return .returnPair (← expr fuel a.toSyntax) (← expr fuel b.toSyntax)
    | _ => .error "unsupported StreamK source statement"

/-- Decoding is successful only if every source statement and helper binding
is accepted. No execution bound or whole-kernel callback is introduced. -/
def decode (source : CS.Program) : Except String (List Statement) :=
  source.mapM (statement 64)
end VeriTile.Bench.Repairs.StreamK.TileChecked.SourceAdapter
