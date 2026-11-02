import VeriTile.Triton.DSL.StatementJitFunctions

/-! Source integer promotion for closed statement helpers. Track uint8 loads
from declared pointer storage, casts, aliases and elementwise expressions.
Unsigned arithmetic wraps after each operation; an integer scalar in `where`
inherits the other branch's signed32 tensor type. These are implicit source
conversions, inserted only in elaboration, following Triton's scalar rules:
https://triton-lang.org/main/python-api/triton-semantics.html#type-promotion -/
namespace VeriTile.Triton.DSL.JitIntegerPromotion
open Lean

inductive Kind where
  | byte | signed32
  deriving BEq
abbrev Facts := List (Name × Kind)

private partial def scalar? (value : TSyntax `tritonExpr) : Option Int :=
  match value with
  | `(tritonExpr| $n:num)=>some n.getNat
  | `(tritonExpr| ($e:tritonExpr))=>scalar? e
  | `(tritonExpr| -$e:tritonExpr)=>(scalar? e).map Int.neg
  | _=>none

private def castKind (dtype : TSyntax `tritonDType) : Option Kind :=
  match dtype with
  | `(tritonDType| tl.uint8)=>some .byte
  | `(tritonDType| tl.int32)=>some .signed32
  | _=>none

private def combine (a b : Option Kind) (left right : TSyntax `tritonExpr) : Option Kind := do
  match a,b with
  | some a,some b=>if a==b then some a else none
  | some a,none=>if (scalar? right).isSome then some a else none
  | none,some b=>if (scalar? left).isSome then some b else none
  | _,_=>none

/-- Only the outer pointer base determines load storage. A nested byte load
in a float pointer's offset does not make the outer load a byte load. -/
private partial def pointerRoot? (value : TSyntax `tritonExpr) : Option Name :=
  match value with
  | `(tritonExpr| $id:ident)=>some id.getId.eraseMacroScopes
  | `(tritonExpr| ($inner:tritonExpr))=>pointerRoot? inner
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $base:tritonExpr - $_:tritonExpr)=>pointerRoot? base
  | _=>none

partial def kind? (facts : Facts) (bytePointers : List Name) (value : TSyntax `tritonExpr) : Option Kind :=
  if let some (_,dtype):=methodCast? value then castKind dtype else
  match value with
  | `(tritonExpr| $id:ident)=>(facts.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| ($inner:tritonExpr))=>kind? facts bytePointers inner
  | `(tritonExpr| tl.cast($_:tritonExpr, $dtype:tritonDType))=>castKind dtype
  | `(tritonExpr| tl.load($pointer:tritonExpr $[, $_:tritonMemKwarg]*))=>
    if (pointerRoot? pointer).any bytePointers.contains then some .byte else none
  | `(tritonExpr| $left:tritonExpr << $_:tritonExpr)
  | `(tritonExpr| $left:tritonExpr >> $_:tritonExpr)=>kind? facts bytePointers left
  | `(tritonExpr| $left:tritonExpr + $right:tritonExpr)
  | `(tritonExpr| $left:tritonExpr - $right:tritonExpr)
  | `(tritonExpr| $left:tritonExpr * $right:tritonExpr)
  | `(tritonExpr| $left:tritonExpr & $right:tritonExpr)
  | `(tritonExpr| $left:tritonExpr | $right:tritonExpr)
  | `(tritonExpr| $left:tritonExpr ^ $right:tritonExpr)
  | `(tritonExpr| tl.interleave($left:tritonExpr, $right:tritonExpr))
  | `(tritonExpr| tl.where($_:tritonExpr, $left:tritonExpr, $right:tritonExpr))=>
    combine (kind? facts bytePointers left) (kind? facts bytePointers right) left right
  | _=>none

private def setFact (facts : Facts) (name : Name) (kind : Option Kind) : Facts :=
  let rest:=facts.filter (·.1!=name)
  match kind with | none=>rest | some value=>(name,value)::rest

/-- Read facts at the actual call point; a later redefinition cannot supply
an earlier helper argument's type. Unknown pointer aliases are not inferred. -/
def byteParameters (bytePointers : List Name) (helper : Name) (parameters : List Name)
    (body : Array (TSyntax `tritonStmt)) : List (List Name) := Id.run do
  let mut facts:Facts:=[]
  let mut available:=bytePointers
  let mut calls:=[]
  for stmt in body do
    if let `(tritonStmt| $_:ident = $callee:ident($args:statementJitArgument,*)):=stmt then
      if callee.getId.eraseMacroScopes==helper then
        let mut supplied:=[]
        let mut index:=0
        for arg in args.getElems do
          match arg with
          | `(statementJitArgument| $name:ident = $value:tritonExpr)=>
            if kind? facts available value==some .byte then supplied:=name.getId.eraseMacroScopes::supplied
          | `(statementJitArgument| $value:tritonExpr)=>
            if let some name:=parameters[index]? then
              if kind? facts available value==some .byte then supplied:=name::supplied
            index:=index+1
          | _=>pure ()
        calls:=calls++[supplied]
    if let `(tritonStmt| $name:ident = $value:tritonExpr):=stmt then
      facts:=setFact facts name.getId.eraseMacroScopes (kind? facts available value)
      available:=available.filter (·!=name.getId.eraseMacroScopes)
  return calls

private partial def lower (facts : Facts) (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (lower facts))
  let expression:TSyntax `tritonExpr:=⟨raw⟩
  if let `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, $no:tritonExpr)):=expression then
    let cast (literal : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
      let some value:=scalar? literal | Macro.throwError "signed where promotion requires a literal"
      unless -(2^31 : Int)≤value && value<2^31 do
        Macro.throwError "weak integer where branch does not fit the signed32 tensor"
      `(tritonExpr| tl.cast($literal:tritonExpr, tl.int32))
    if kind? facts [] no==some .signed32 && (scalar? yes).isSome then
      let yes ← cast yes
      return (← `(tritonExpr| tl.where($condition, $yes, $no))).raw
    if kind? facts [] yes==some .signed32 && (scalar? no).isSome then
      let no ← cast no
      return (← `(tritonExpr| tl.where($condition, $yes, $no))).raw
  if let `(tritonExpr| $left:tritonExpr << $_:tritonExpr):=expression then
    if kind? facts [] left==some .byte then
      Macro.throwError "uint8 left shifts require an explicit source-width lowering"
  -- Python scalars outside this bounded subset need a different promotion
  -- rule. Reject them rather than silently narrowing a wide/negative scalar.
  let operands:=match expression with
    | `(tritonExpr| $left:tritonExpr + $right:tritonExpr)
    | `(tritonExpr| $left:tritonExpr - $right:tritonExpr)
    | `(tritonExpr| $left:tritonExpr * $right:tritonExpr)
    | `(tritonExpr| $left:tritonExpr & $right:tritonExpr)
    | `(tritonExpr| $left:tritonExpr | $right:tritonExpr)
    | `(tritonExpr| $left:tritonExpr ^ $right:tritonExpr)=>some (left,right)
    | _=>none
  if let some (left,right):=operands then
    for (tensor,literal) in [(left,right),(right,left)] do
      if kind? facts [] tensor==some .byte then
        if let some n:=scalar? literal then
          unless 0≤n && n<256 do
            Macro.throwError "uint8 weak scalar outside the supported range 0..255"
        else unless kind? facts [] literal==some .byte do
          Macro.throwError "uint8 arithmetic requires another byte tensor or a closed weak integer"
  let operation:=match expression with
    | `(tritonExpr| $left:tritonExpr + $right:tritonExpr)=>some (left,right,0)
    | `(tritonExpr| $left:tritonExpr - $right:tritonExpr)=>some (left,right,1)
    | `(tritonExpr| $left:tritonExpr * $right:tritonExpr)=>some (left,right,2)
    | _=>none
  if let some (left,right,operator):=operation then
    if kind? facts [] expression==some .byte then
      let left ← `(tritonExpr| tl.cast($left:tritonExpr, tl.int64))
      let right ← `(tritonExpr| tl.cast($right:tritonExpr, tl.int64))
      let widened ← if operator==0 then `(tritonExpr| $left + $right)
        else if operator==1 then `(tritonExpr| $left - $right)
        else `(tritonExpr| $left * $right)
      return (← `(tritonExpr| tl.cast($widened:tritonExpr, tl.uint8))).raw
  return raw

def normalize (byteInputs : List Name) (body : Array Syntax) : MacroM (Array Syntax) := do
  let mut facts:Facts:=byteInputs.map (·,.byte)
  let mut out:=#[]
  for raw in body do
    let original:TSyntax `tritonStmt:=⟨raw⟩
    let normalized ← lower facts raw
    out:=out.push normalized
    if let `(tritonStmt| $name:ident = $value:tritonExpr):=original then
      facts:=setFact facts name.getId.eraseMacroScopes (kind? facts [] value)
  return out

end VeriTile.Triton.DSL.JitIntegerPromotion
