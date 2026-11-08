import VeriTile.Triton.DSL.PythonFullConstructors

/-! Source typing for packed integer unpacking. Read-only literal aliases
feeding shift counts keep integer carriers. A low-bit mask bounds its signed32/signed64 integer
result below 2^24. Sums/differences retain a bound only while the total
absolute bound remains below 2^24; only that subset receives the implicit conversion
when multiplied by a known floating tensor. Conversion follows that tensor's
actual dtype, including half/bfloat storage. Unknown combinations remain for
the ordinary checked elaborator to reject. No pasted statement is removed. -/
namespace VeriTile.Triton.DSL.PackedIntegerPromotion
open Lean Elab Term Meta
syntax "triton.cdiv(" tritonExpr "," tritonExpr ")" : tritonExpr

inductive Kind where
  | unknown | integer | bounded (upper : Nat) | floating | pointer (floating : Bool)
  deriving BEq, Inhabited
abbrev Facts := List (Name × Kind)
private def fact (facts : Facts) (name : Name) : Kind :=
  (facts.find? (·.1==name.eraseMacroScopes)).map Prod.snd |>.getD .unknown
private def set (facts : Facts) (name : Name) (value : Kind) : Facts :=
  (name,value)::facts.filter (·.1!=name)
private def integral : Kind → Bool
  | .integer | .bounded _ => Bool.true | _ => Bool.false
private partial def choose (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then choose raw[0]!
  else raw.setArgs (raw.getArgs.map choose)
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def assignments (raw : Syntax) : List (Name × Syntax) :=
  let own := match (⟨raw⟩ : TSyntax `tritonStmt) with
    | `(tritonStmt| $id:ident = $v:tritonExpr) => [(id.getId.eraseMacroScopes,v.raw)]
    | _ => []
  own ++ raw.getArgs.toList.flatMap assignments
private partial def writes (raw : Syntax) : List Name := Id.run do
  let args:=raw.getArgs.toList
  let mut out:=args.flatMap writes
  for op in ["=",":=","+=","-=","*=","/=","//=","%=","&=","|=","^="] do
    if let some i:=args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==op) then
      out:=out++(args.take i).flatMap ids
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString && counter.isIdent then
      out:=counter.getId.eraseMacroScopes::out
  return out
private partial def shiftNames (raw : Syntax) : List Name :=
  let own:=match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| $_:tritonExpr >> $count:tritonExpr)
    | `(tritonExpr| $_:tritonExpr << $count:tritonExpr) => ids count.raw
    | _ => []
  own++raw.getArgs.toList.flatMap shiftNames
private def integerAliases (body : Array (TSyntax `tritonStmt)) : List Name := Id.run do
  let bindings:=body.toList.flatMap (assignments ·.raw)
  let assigned:=body.toList.flatMap (writes ·.raw)
  let mut needed:=(body.toList.flatMap (shiftNames ·.raw)).eraseDups
  for _ in [:bindings.length+1] do
    needed:=(needed++(bindings.filter (fun p=>needed.contains p.1)).flatMap (ids ·.2)).eraseDups
  return needed.filter fun name=>assigned.count name==1 && bindings.any (fun p=>
    p.1==name && match (⟨p.2⟩ : TSyntax `tritonExpr) with
      | `(tritonExpr| $n:num)=>n.getNat<2^31
      | _=>Bool.false)

/-- A source constexpr assignment keeps its readable binding and a constant
carrier in the IR. Only declared Nat constexprs and previous aliases may feed it. -/
private partial def constantExpression (constants : List Name)
    (aliases : List (Name × TSyntax `term)) (value : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match value with
  | `(tritonExpr| $n:num)=>`(term| ($n:num : Nat))
  | `(tritonExpr| $name:ident)=>
    if let some (_,value):=aliases.find? (·.1==name.getId.eraseMacroScopes) then return value
    unless constants.contains name.getId.eraseMacroScopes do
      Macro.throwErrorAt name "integer constexpr assignment requires declared constexpr inputs"
    `(term| ($name:ident : Nat))
  | `(tritonExpr| ($value:tritonExpr))=>constantExpression constants aliases value
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)=>do
    let a ← constantExpression constants aliases a; let b ← constantExpression constants aliases b
    `(term| $a + $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)=>do
    let a ← constantExpression constants aliases a; let b ← constantExpression constants aliases b
    `(term| $a * $b)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)=>do
    let a ← constantExpression constants aliases a; let b ← constantExpression constants aliases b
    `(term| $a / $b)
  | `(tritonExpr| triton.cdiv($a:tritonExpr, $b:tritonExpr))
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr))=>do
    let a ← constantExpression constants aliases a; let b ← constantExpression constants aliases b
    `(term| (($a + $b - 1) / $b))
  | _=>Macro.throwErrorAt value "unsupported integer constexpr assignment"
private def constantAssignments (constants : List Name) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let assigned:=body.toList.flatMap (writes ·.raw)
  let mut future:=body.toList.filterMap fun st=>match st with
    | `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr)=>some name.getId.eraseMacroScopes
    | _=>none
  let mut aliases:List (Name × TSyntax `term):=[]
  let mut result:=#[]
  for st in body do
    match st with
    | `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr)=>
      let key:=name.getId.eraseMacroScopes
      unless assigned.count key==1 && !constants.contains key do
        Macro.throwErrorAt name "constexpr assignment must have one unshadowed binding"
      let value ← constantExpression constants aliases value
      aliases:=aliases++[(key,value)]
      future:=future.filter (·!=key)
      result:=result.push (← `(tritonStmt| $name:ident = $($value:term)))
    | _=>
      if (ids st.raw).any future.contains then
        Macro.throwErrorAt st "constexpr assignment used before its binding"
      result:=result.push st
  return result

private partial def viewBase (e : TSyntax `tritonExpr) : Option Ident :=
  match e with
  | `(tritonExpr| $id:ident)=>some id
  | `(tritonExpr| ($e:tritonExpr))=>viewBase e
  | `(tritonExpr| $e:tritonExpr[None, :])
  | `(tritonExpr| $e:tritonExpr[:, None])
  | `(tritonExpr| $e:tritonExpr[:])=>viewBase e
  | _=>none
private def castDType? (e : TSyntax `tritonExpr) : Option (TSyntax `tritonDType) :=
  (methodCast? e).map Prod.snd <|> (match e with
    | `(tritonExpr| tl.cast($_:tritonExpr, $dt:tritonDType))=>some dt
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*).to($dt:tritonDType))=>some dt
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($dt:tritonDType))=>some dt
    | _=>none)
private partial def kind (facts : Facts) (e : TSyntax `tritonExpr) : Kind :=
  if let some dtype:=castDType? e then
    match dtype with
    | `(tritonDType| tl.int32) | `(tritonDType| tl.int64)=>.integer
    | `(tritonDType| tl.int1) | `(tritonDType| tl.int8) | `(tritonDType| tl.int16)
    | `(tritonDType| tl.uint8) | `(tritonDType| tl.uint16)
    | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64)=>.unknown
    | _=>.floating
  else match e with
  | `(tritonExpr| $id:ident)=>fact facts id.getId
  | `(tritonExpr| $_:num)=>.integer
  | `(tritonExpr| ($e:tritonExpr))=>kind facts e
  | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*))
  | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*))=>
    match kind facts p with | .pointer Bool.true=>.floating | .pointer Bool.false=>.integer | _=>.unknown
  | `(tritonExpr| tl.program_id($_:tritonExpr))
  | `(tritonExpr| tl.program_id(axis=$_:num))
  | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr))=>.integer
  | `(tritonExpr| tl.toReal($_:tritonExpr))
  | `(tritonExpr| tl.dot($_:tritonExpr, $_:tritonExpr))=>.floating
  | `(tritonExpr| $e:tritonExpr[None, :])
  | `(tritonExpr| $e:tritonExpr[:, None])=>kind facts e
  | `(tritonExpr| $left:tritonExpr & $mask:num)=>
    if integral (kind facts left) && mask.getNat<2^24 then .bounded mask.getNat else .unknown
  | `(tritonExpr| $left:tritonExpr >> $_:tritonExpr)
  | `(tritonExpr| $left:tritonExpr << $_:tritonExpr)=>
    if integral (kind facts left) then .integer else .unknown
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)=>
    let ka:=kind facts a; let kb:=kind facts b
    match ka,kb with
    | .bounded x,.bounded y=>if x+y<2^24 then .bounded (x+y) else .unknown
    | .pointer f,_=>.pointer f
    | _,.pointer f=>.pointer f
    | .floating,_ | _,.floating=>.floating
    | _,_=>if integral ka && integral kb then .integer else .unknown
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr)=>
    let ka:=kind facts a; let kb:=kind facts b
    match ka,kb with
    | .pointer f,_=>.pointer f
    | _,.pointer f=>.pointer f
    | .floating,_ | _,.floating=>.floating
    | _,_=>if integral ka && integral kb then .integer else .unknown
  | _=>.unknown

private partial def promote (facts : Facts) (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (promote facts))
  if let `(tritonExpr| $left:tritonExpr * $right:tritonExpr):=raw then
    for (integer,floating,flipped) in [(left,right,Bool.false),(right,left,Bool.true)] do
      if let .bounded _:=kind facts integer then
        if kind facts floating==.floating then
          let some base:=viewBase floating | continue
          let target:=mkIdent (Name.str base.getId.eraseMacroScopes "dtype")
          let converted ← `(tritonExpr| (tl.toReal($integer:tritonExpr)).to($target:ident))
          return (← if flipped then `(tritonExpr| $floating * $converted)
            else `(tritonExpr| $converted * $floating)).raw
  return raw

private def merge (before yes no : Facts) : Facts :=
  ((before.map Prod.fst++yes.map Prod.fst++no.map Prod.fst).eraseDups).map fun name=>
    let a:=fact yes name; let b:=fact no name
    (name, if a==b then a else if a==.unknown then b else if b==.unknown then a else .unknown)

private partial def lower (naturals : List Name) (facts : Facts)
    (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt) × Facts) := do
  let mut facts:=facts
  let mut result:=#[]
  for original in body do
    let st:TSyntax `tritonStmt:=⟨choose original.raw⟩
    let next ← match st with
    | `(tritonStmt| $id:ident = $value:tritonExpr)=>do
      let value ← if naturals.contains id.getId.eraseMacroScopes then
        match value with
        | `(tritonExpr| $n:num)=>`(tritonExpr| $($n:num))
        | _=>pure value
        else pure value
      let value:TSyntax `tritonExpr:=⟨← promote facts value.raw⟩
      facts:=set facts id.getId.eraseMacroScopes (kind facts value)
      `(tritonStmt| $id:ident = $value:tritonExpr)
    | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>do
      let (yes,yfacts) ← lower naturals facts yes
      let (no,nfacts) ← lower naturals facts no
      facts:=merge facts yfacts nfacts
      `(tritonStmt| if $c:tritonExpr { $yes* } else { $no* })
    | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* })=>do
      let (yes,yfacts) ← lower naturals facts yes
      facts:=merge facts yfacts facts
      `(tritonStmt| if $c:tritonExpr { $yes* })
    | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr) { $loop:tritonStmt* })=>do
      let (loop,after) ← lower naturals (set facts i.getId.eraseMacroScopes .integer) loop
      facts:=merge facts after facts
      `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr) { $loop* })
    | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $step:tritonExpr) { $loop:tritonStmt* })=>do
      let (loop,after) ← lower naturals (set facts i.getId.eraseMacroScopes .integer) loop
      facts:=merge facts after facts
      `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $step:tritonExpr) { $loop* })
    | _=>pure (⟨← promote facts st.raw⟩ : TSyntax `tritonStmt)
    result:=result.push next
  return (result,facts)

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "packed source metadata requires static lists"
private def prepare (stx : Syntax) : TermElabM Syntax := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut facts:Facts:=[]
  for projection in [``Triton.CallSignature.pointers,``Triton.CallSignature.floatingPointers] do
    for entry in ← listValues (← mkAppM projection #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry]) | throwError "pointer names must be literal"
      let dtype ← whnf (← mkAppM ``Prod.snd #[entry])
      let floating:=projection==``Triton.CallSignature.floatingPointers ||
        [``Triton.SourceDType.fp16,``Triton.SourceDType.bf16,``Triton.SourceDType.fp32,
         ``Triton.SourceDType.fp64,``Triton.SourceDType.f8e4,``Triton.SourceDType.f8e5].any dtype.isConstOf
      let signedWide:=[``Triton.SourceDType.i32,``Triton.SourceDType.i64].any dtype.isConstOf
      facts:=set facts name.toName (if floating || signedWide then .pointer floating else .unknown)
  let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM fun value=>do
    let .lit (.strVal name) ← whnf value | throwError "constexpr names must be literal"
    return name.toName
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure (choose parsed) | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete packed source required"
  let body ← liftMacroM (constantAssignments constants body)
  let body ← liftMacroM (lower (integerAliases body) facts body)
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (body.1.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  CallSignature.elaborateInvocation (← prepare stx) expected
end VeriTile.Triton.DSL.PackedIntegerPromotion
