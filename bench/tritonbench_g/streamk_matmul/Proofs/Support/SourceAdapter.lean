import VeriTile.Triton.DSL.PastedControlFunctions
import bench.tritonbench_g.streamk_matmul.Proofs.Support.SourceReader

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

private def methodCast (raw : Syntax) : Option (Syntax × Syntax) :=
  let k:=raw.getKind
  let args:=raw.getArgs
  if k==``VeriTile.Triton.DSL.tritonDottedIdentMethodCast && args.size==4 then
    match args[0]!.getId.eraseMacroScopes with
    | .str base "to" => some (.ident SourceInfo.none "".toRawSubstring base [],args[2]!)
    | _ => none
  else if k==``VeriTile.Triton.DSL.tritonMethodCast ||
      k==``VeriTile.Triton.DSL.tritonIdentMethodCast ||
      k==``VeriTile.Triton.DSL.tritonIdentMethodCastSpaced then
    if args.size==5 then some (args[0]!,args[3]!)
    else if args.size==6 then some (args[0]!,args[4]!) else none
  else none

private def expr : Nat → Syntax → Except String Reader.Expr
  | 0, _ => .error "integer expression decoding depth exceeded"
  | fuel + 1, raw => do
    if raw.isIdent then return .variable (identifier raw.getId.eraseMacroScopes)
    else if let some (operand,dtype) := methodCast raw then
      match (⟨dtype⟩ : TSyntax `tritonDType) with
      | `(tritonDType| tl.int64) => return .signed64 (← expr fuel operand)
      | _ => .error "unsupported StreamK integer method cast"
    else match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| ($x:tritonExpr)) => expr fuel x.raw
    | `(tritonExpr| end) => .ok (.variable "end")
    | `(tritonExpr| $n:num) => return .literal (← numeral n.raw)
    | `(tritonExpr| tl.program_id(0)) => .ok .pid
    | `(tritonExpr| tl.cast($x:tritonExpr, tl.int64))
    | `(tritonExpr| tl.full((), $x:tritonExpr, tl.int64))
    | `(tritonExpr| tl.full([], $x:tritonExpr, tl.int64)) => return .signed64 (← expr fuel x.raw)
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
  if let some (operand,dtype) := methodCast raw then
    match (⟨dtype⟩ : TSyntax `tritonDType), (⟨operand⟩ : TSyntax `tritonExpr) with
    | `(tritonDType| tl.float16), `(tritonExpr| $y:ident) =>
      if target != name y then .error "unsupported conversion target" else .ok (.half target)
    | _,_ =>
      if operand.isIdent && target==identifier operand.getId.eraseMacroScopes then
        match (⟨dtype⟩ : TSyntax `tritonDType) with
        | `(tritonDType| tl.float16) => .ok (.half target)
        | _ => .error "unsupported source assignment cast"
      else return .integer target (← expr fuel raw)
  else match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.zeros([$m:tritonExpr, $n:tritonExpr], dtype=tl.float32))
  | `(tritonExpr| tl.zeros([$m:tritonExpr, $n:tritonExpr], dtype=ACC_TYPE)) =>
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

private def tileCall (helpers : List (String × String)) (fuel : Nat) (targets : List String) (function : String)
    (args : List CS.Argument) : Except String Statement := do
  if targets != ["pid_m","pid_n"] then .error "unsupported mapping outputs" else
    let args ← positional args
    match args with
    | tile :: tail =>
      checkArgs tail ["M","N","K","BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M"]
      let swizzle ← match helpers.lookup function with
        | some "swizzle" => .ok Bool.true
        | some "linear" => .ok Bool.false
        | _ => .error "unsupported mapping helper"
      return .tileCall swizzle (← expr fuel tile.toSyntax)
    | [] => .error "missing mapping arguments"

private def effect (helpers : List (String × String)) (fuel : Nat) (function : String) (args : List CS.Argument) : Except String Statement := do
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
  | callee, _ =>
    unless helpers.lookup callee==some "mac" do throw ("unsupported StreamK source call: "++callee)
    let args ← positional args
    checkArgs args ["A","B","C","M","N","K","partials","pid","stride_am","stride_ak",
      "stride_bk","stride_bn","stride_cm","stride_cn","iters_per_tile","start_iter","end_iter",
      "BLOCK_M","BLOCK_N","BLOCK_K","ACC_TYPE","GROUP_M"]
    .ok .macCall

private def statement (helpers : List (String × String)) : Nat → CS.Statement → Except String Statement
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
    | .bindCall targets function args => tileCall helpers fuel targets function args
    | .call function args => effect helpers fuel function args
    | .branch condition yes no =>
      return .branch (← test fuel condition.toSyntax)
        (← yes.mapM (statement helpers fuel)) (← no.mapM (statement helpers fuel))
    | .range index [lo,hi] body =>
      return .range index (← expr fuel lo.toSyntax) (← expr fuel hi.toSyntax) (← body.mapM (statement helpers fuel))
    | .whileLoop condition body => return .loop (← test fuel condition.toSyntax) (← body.mapM (statement helpers fuel))
    | .returnValues [a,b] => return .returnPair (← expr fuel a.toSyntax) (← expr fuel b.toSyntax)
    | _ => .error "unsupported StreamK source statement"

private def dtypeTag : VeriTile.Triton.SourceDType → String
  | .fp32 => "fp32" | _ => "unsupported"
private def scalarTag : VeriTile.Triton.ScalarType → String
  | .integer => "integer" | _ => "unsupported"
private def role : String → Option String
  | "swizzle_tile" | "swizzle_tile_area" => some "swizzle"
  | "linear_tile" | "linear_tile_area" => some "linear"
  | "mac_loop" | "mac_loop_area" => some "mac"
  | "first_wave" | "first_wave_area" => some "first"
  | "full_tiles" | "full_tiles_area" => some "full"
  | "reduce_streamk" | "reduce_streamk_area" => some "reduce"
  | _ => none

private structure ABI where
  pointers : List (String × String)
  scalars : List (String × String)
  constants : List String
  reads : List String
  writes : List String

private def abi : String → Option ABI
  | "swizzle" => some ⟨[],[("tile_id","integer"),("M","integer"),("N","integer"),("K","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"],[],[]⟩
  | "linear" => some ⟨[],[("tile_id","integer"),("M","integer"),("N","integer"),("K","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"],[],[]⟩
  | "mac" => some ⟨[("A","fp32"),("B","fp32"),("C","fp32"),("partials","fp32")],[("M","integer"),("N","integer"),("K","integer"),("program","integer"),("stride_am","integer"),("stride_ak","integer"),("stride_bk","integer"),("stride_bn","integer"),("stride_cm","integer"),("stride_cn","integer"),("iters_per_tile","integer"),("start_iter","integer"),("end_iter","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"],["A", "B"],["C", "partials"]⟩
  | "first" => some ⟨[("A","fp32"),("B","fp32"),("C","fp32"),("partials","fp32")],[("M","integer"),("N","integer"),("K","integer"),("stride_am","integer"),("stride_ak","integer"),("stride_bk","integer"),("stride_bn","integer"),("stride_cm","integer"),("stride_cn","integer"),("total_full_tiles_streamk","integer"),("total_partial_tiles_streamk","integer"),("iters_per_tile","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"],["A", "B"],["C", "partials"]⟩
  | "full" => some ⟨[("A","fp32"),("B","fp32"),("C","fp32")],[("M","integer"),("N","integer"),("K","integer"),("stride_am","integer"),("stride_ak","integer"),("stride_bk","integer"),("stride_bn","integer"),("stride_cm","integer"),("stride_cn","integer"),("total_tiles_streamk","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "ACC_TYPE", "GROUP_M"],["A", "B"],["C"]⟩
  | "reduce" => some ⟨[("C","fp32"),("partials","fp32")],[("M","integer"),("N","integer"),("K","integer"),("stride_cm","integer"),("stride_cn","integer"),("total_programs_streamk","integer"),("total_full_tiles_streamk","integer"),("total_partial_tiles_streamk","integer"),("iters_per_tile","integer")],["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"],["C", "partials"],["C", "partials"]⟩
  | _ => none

/-- Check the caller dtype and complete ABI before interpreting source forms.
The ACC_TYPE token is specialized only from this checked FP32 caller binding. -/
def profileSupported (source : VeriTile.Triton.DSL.PastedControlFunctions.Invocation) : Bool :=
  match role source.entrypoint with
  | none => false
  | some kind => match abi kind with
    | none => false
    | some expected =>
      let p:=source.signature
      p.pointers.map (fun (n,t)=>(n,dtypeTag t))==expected.pointers &&
      p.scalars.map (fun (n,t)=>(n,scalarTag t))==expected.scalars &&
      p.constexpr==expected.constants && p.readPorts==expected.reads && p.writePorts==expected.writes &&
      p.floatingPointers.isEmpty && p.sourceConstants.isEmpty && p.unaryImports.isEmpty &&
      source.constexprBools.isEmpty &&
      source.dtypeConstants.map (fun (n,t)=>(n,dtypeTag t))==
        (if expected.constants.contains "ACC_TYPE" then [("ACC_TYPE","fp32")] else []) &&
      source.helperBindings.map Prod.snd==
        (if kind=="mac" || kind=="full" || kind=="reduce" then ["swizzle","linear"]
         else if kind=="first" then ["mac"] else [])

/-- Every statement is decoded; named helper calls must use the checked caller
binding. No loop bound or whole-kernel callback is introduced. -/
def decode (source : VeriTile.Triton.DSL.PastedControlFunctions.Invocation) : Except String (List Statement) := do
  unless profileSupported source do throw "unsupported StreamK caller ABI, dtype or helper binding"
  source.body.mapM (statement source.helperBindings 64)

/-- Link calls to the actual supplied source members before producing Sources.
This checks helper identities, not just compatible arity or an enum name. -/
def decodeSources (swizzle linear mac first fullKernel reduce : VeriTile.Triton.DSL.PastedControlFunctions.Invocation) :
    Except String Sources := do
  unless swizzle.helperBindings.isEmpty && linear.helperBindings.isEmpty &&
    mac.helperBindings==[(swizzle.entrypoint,"swizzle"),(linear.entrypoint,"linear")] &&
    first.helperBindings==[(mac.entrypoint,"mac")] &&
    fullKernel.helperBindings==[(swizzle.entrypoint,"swizzle"),(linear.entrypoint,"linear")] &&
    reduce.helperBindings==[(swizzle.entrypoint,"swizzle"),(linear.entrypoint,"linear")] do
    throw "StreamK helper does not link to the supplied source member"
  return ⟨← decode swizzle,← decode linear,← decode mac,← decode first,← decode fullKernel,← decode reduce⟩
end VeriTile.Bench.Repairs.StreamK.TileChecked.SourceAdapter
