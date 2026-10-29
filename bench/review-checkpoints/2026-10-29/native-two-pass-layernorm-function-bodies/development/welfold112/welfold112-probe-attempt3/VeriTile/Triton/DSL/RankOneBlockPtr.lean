import VeriTile.Triton.DSL.PythonBody
import Lean

/-! Lower the positional rank-one block-pointer subset to signed element
addresses and captured logical bounds. The source quotation stays intact.
Coordinates and parent shape are captured at construction, so subsequent
register writes cannot change a previously constructed pointer. Unsupported
pointer uses and escaping control-flow bindings fail closed. -/
namespace VeriTile.Triton.DSL.RankOneBlockPtr
open Lean Elab Term Meta

private structure Pointer where
  name : Name
  region : Ident
  coordinates : Ident
  parentShape : Ident

private partial def names (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap names

private def axisZero (axisIndex : TSyntax `tritonExpr) : TermElabM Unit := do
  let value ← match axisIndex with
    | `(tritonExpr| $n:num) => pure n.getNat
    | `(tritonExpr| $(($t:term : Nat))) => do
        let e ← elabTerm t (some (mkConst ``Nat))
        let some n ← getNatValue? (← whnf e)
          | throwErrorAt axisIndex "rank-one block-pointer axis must be a closed constexpr"
        pure n
    | _ => throwErrorAt axisIndex "rank-one block-pointer axis must be a closed constexpr"
  unless value == 0 do throwErrorAt axisIndex "rank-one block-pointer axis must be zero"

private partial def base (value : TSyntax `tritonExpr) :
    MacroM (Ident × TSyntax `tritonExpr) := do
  match value with
  | `(tritonExpr| ($inner:tritonExpr)) => base inner
  | `(tritonExpr| $region:ident) =>
      return (region, ← `(tritonExpr| $((0 : Int))))
  | `(tritonExpr| $region:ident + $offset:tritonExpr) =>
      return (region, offset)
  | _ => Macro.throwErrorAt value "rank-one block pointer requires a named region and an optional element offset"

/-- Read only the block-specific options; other kwargs retain their original
source spelling for the ordinary memory compiler. -/
private def memoryOptions (kwargs : Array (TSyntax `tritonMemKwarg)) (loading : Bool) :
    TermElabM (Bool × Bool × Array (TSyntax `tritonMemKwarg)) := do
  let mut checked := Bool.false
  let mut zero := Bool.false
  let mut padding := Bool.false
  let mut rest := #[]
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| boundary_check=($axisIndex:tritonExpr,)) =>
        if checked then throwErrorAt kw "duplicate boundary_check"
        axisZero axisIndex
        checked := Bool.true
    | `(tritonMemKwarg| padding_option=$value:str) =>
        unless loading do throwErrorAt kw "padding_option is only valid for loads"
        if padding then throwErrorAt kw "duplicate padding_option"
        padding := Bool.true
        unless ["","zero"].contains value.getString do
          throwErrorAt kw "rank-one block-pointer padding supports undefined or zero"
        zero := value.getString == "zero"
    | `(tritonMemKwarg| $key:ident = $_:tritonExpr) =>
        if ["mask","other","boundary_check","padding_option"].contains key.getId.toString then
          throwErrorAt kw "block pointers cannot use ordinary mask/other options"
        rest := rest.push kw
    | _ => rest := rest.push kw
  if zero && !checked then
    throwError "rank-one block-pointer padding needs an explicit boundary_check"
  return (checked,zero,rest)

private def find (pointers : List Pointer) (id : Ident) : Option Pointer :=
  pointers.find? (·.name == id.getId.eraseMacroScopes)

private def access (pointer : Pointer) (id : Ident) (checked : Bool) :
    MacroM (TSyntax `tritonExpr × Array (TSyntax `tritonMemKwarg)) := do
  let address ← `(tritonExpr| $pointer.region:ident + $id:ident)
  if !checked then return (address,#[])
  let coords := pointer.coordinates
  let parent := pointer.parentShape
  let maskName := mkIdent `mask
  let mask ← `(tritonMemKwarg| $maskName:ident=(($((0 : Int)) <= $coords:ident) & ($coords:ident < $parent:ident)))
  return (address,#[mask])

private partial def rewrite (pointers : List Pointer) (allPointers : List Name)
    (raw : Syntax) : TermElabM Syntax := do
  -- Consume a pointer use before recursively visiting its identifier.
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  match expression with
  | `(tritonExpr| tl.load($id:ident $[, $kwargs:tritonMemKwarg]*)) =>
      if let some pointer := find pointers id then
        let (checked,zero,rest) ← memoryOptions kwargs Bool.true
        let (address,mask) ← liftMacroM <| access pointer id checked
        let rest := mask ++ rest
        let rest ← if zero then do
            let otherName := mkIdent `other
            let other ← liftMacroM `(tritonMemKwarg| $otherName:ident=0)
            pure (rest.push other)
          else pure rest
        return (← liftMacroM `(tritonExpr| tl.load($address $[, $rest:tritonMemKwarg]*))).raw
  | _ => pure ()
  let statement : TSyntax `tritonStmt := ⟨raw⟩
  let storeValue (id : Ident) (value : TSyntax `tritonExpr)
      (kwargs : Array (TSyntax `tritonMemKwarg)) : TermElabM (Option Syntax) := do
    if let some pointer := find pointers id then
      let (checked,_,rest) ← memoryOptions kwargs Bool.false
      let (address,mask) ← liftMacroM <| access pointer id checked
      let value : TSyntax `tritonExpr := ⟨← rewrite pointers allPointers value.raw⟩
      let rest := mask ++ rest
      if rest.isEmpty then
        return some (← liftMacroM `(tritonStmt| tl.store($address, $value))).raw
      else
        let first := rest[0]!
        let more := rest.extract 1 rest.size
        return some (← liftMacroM `(tritonStmt| tl.store($address, $value, $first:tritonMemKwarg $[, $more:tritonMemKwarg]*))).raw
    return none
  match statement with
  | `(tritonStmt| tl.store($id:ident, $value:tritonExpr)) =>
      if let some result ← storeValue id value #[] then return result
  | `(tritonStmt| tl.store($id:ident, $value:tritonExpr, $first:tritonMemKwarg $[, $more:tritonMemKwarg]*)) =>
      if let some result ← storeValue id value (#[first] ++ more) then return result
  | _ => pure ()
  -- Retain the dedicated complete load-and-cast production: nesting an
  -- ordinary floating load inside a generic expression loses its load event.
  match expression with
  | `(tritonExpr| tl.load($id:ident $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType)) =>
      if let some pointer := find pointers id then
        let (checked,zero,rest) ← memoryOptions kwargs Bool.true
        let (address,mask) ← liftMacroM <| access pointer id checked
        let rest := mask ++ rest
        let rest ← if zero then do
            let otherName := mkIdent `other
            let other ← liftMacroM `(tritonMemKwarg| $otherName:ident=0)
            pure (rest.push other)
          else pure rest
        return (← liftMacroM `(tritonExpr| tl.load($address $[, $rest:tritonMemKwarg]*).to($dtype:tritonDType))).raw
  | `(tritonExpr| tl.load($id:ident $[, $kwargs:tritonMemKwarg]*).to($dtype:ident)) =>
      if let some pointer := find pointers id then
        let (checked,zero,rest) ← memoryOptions kwargs Bool.true
        let (address,mask) ← liftMacroM <| access pointer id checked
        let rest := mask ++ rest
        let rest ← if zero then do
            let otherName := mkIdent `other
            let other ← liftMacroM `(tritonMemKwarg| $otherName:ident=0)
            pure (rest.push other)
          else pure rest
        let dtype : Ident := ⟨← rewrite pointers allPointers dtype.raw⟩
        return (← liftMacroM `(tritonExpr| tl.load($address $[, $rest:tritonMemKwarg]*).to($dtype:ident))).raw
  | `(tritonExpr| tl.load($id:ident $[, $kwargs:tritonMemKwarg]*).to($owner:ident.$field:ident)) =>
      if let some pointer := find pointers id then
        let (checked,zero,rest) ← memoryOptions kwargs Bool.true
        let (address,mask) ← liftMacroM <| access pointer id checked
        let rest := mask ++ rest
        let rest ← if zero then do
            let otherName := mkIdent `other
            let other ← liftMacroM `(tritonMemKwarg| $otherName:ident=0)
            pure (rest.push other)
          else pure rest
        let owner : Ident := ⟨← rewrite pointers allPointers owner.raw⟩
        return (← liftMacroM `(tritonExpr| tl.load($address $[, $rest:tritonMemKwarg]*).to($owner:ident.$field:ident))).raw
  | _ => pure ()
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes
    if let some pointer := pointers.find? (fun p =>
        [p.name ++ `type,p.name ++ `dtype,
          p.name ++ `type.element_ty,p.name ++ `dtype.element_ty].contains name) then
      return (mkIdent (name.replacePrefix pointer.name pointer.region.getId.eraseMacroScopes)).raw
    if allPointers.contains name then
      throwErrorAt raw "unsupported or escaping rank-one block-pointer use"
    return raw
  match raw with
  | .node info kind args =>
      return .node info kind (← args.mapM (rewrite pointers allPointers))
  | _ => return raw

private partial def lowerBlock (used : List Name) (allPointers : List Name)
    (body : List (TSyntax `tritonStmt)) (pointers : List Pointer) (counter : Nat) :
    TermElabM (List (TSyntax `tritonStmt) × Nat) := do
  match body with
  | [] => return ([],counter)
  | stmt :: rest =>
      let mut pointers := pointers
      let mut counter := counter
      let mut emitted := [stmt]
      match stmt with
      | `(tritonStmt| $id:ident = tl.make_block_ptr($p:tritonExpr, ($parent:tritonExpr,),
          ($stride:tritonExpr,), ($offset:tritonExpr,), ($width:tritonExpr,), $orderKw:ident=($axisIndex:tritonExpr,))) =>
          unless orderKw.getId.eraseMacroScopes.toString == "order" do
            throwErrorAt orderKw "rank-one block pointer expects order="
          axisZero axisIndex
          let fresh (tag : String) : TermElabM Ident := do
            let candidate := Name.mkSimple ("__veritile_block_" ++ id.getId.eraseMacroScopes.toString ++ "_" ++ tag)
            if used.any (fun n => n == candidate || candidate.isPrefixOf n) then
              throwErrorAt stmt "source identifier conflicts with a private block-pointer capture"
            return mkIdent candidate
          let coordinates ← fresh "coordinates"
          let parentShape ← fresh "shape"
          let (region,origin) ← liftMacroM <| base p
          -- Capture signed logical coordinates and bounds before later source
          -- assignments; perform the final address-to-Nat conversion at access.
          let coord ← liftMacroM `(tritonStmt| $coordinates:ident = $((0 : Int)) + ($offset).to(tl.int64) + tl.arange(0, $width).to(tl.int64))
          let shape ← liftMacroM `(tritonStmt| $parentShape:ident = ($parent).to(tl.int64))
          let address ← liftMacroM `(tritonStmt| $id:ident = ($origin).to(tl.int64) + $coordinates:ident * ($stride).to(tl.int64))
          emitted := [coord,shape,address]
          pointers := {name := id.getId.eraseMacroScopes,region,coordinates,parentShape} ::
            pointers.filter (·.name != id.getId.eraseMacroScopes)
          counter := counter + 1
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          let condition : TSyntax `tritonExpr := ⟨← rewrite pointers allPointers condition.raw⟩
          let (yes,n) ← lowerBlock used allPointers yes.toList pointers counter
          let (no,n) ← lowerBlock used allPointers no.toList pointers n
          let yes := yes.toArray
          let no := no.toArray
          emitted := [← liftMacroM `(tritonStmt| if $condition { $yes* } else { $no* })]
          pointers := []
          counter := n
      | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $inner:tritonStmt* }) =>
          let (inner,n) ← lowerBlock used allPointers inner.toList pointers counter
          let inner := inner.toArray
          emitted := [← liftMacroM `(tritonStmt| for $i in range($start, $stop) { $inner* })]
          pointers := []
          counter := n
      | _ =>
          emitted := [⟨← rewrite pointers allPointers stmt.raw⟩]
      let (rest,n) ← lowerBlock used allPointers rest pointers counter
      return (emitted ++ rest,n)

private partial def pointerNames (raw : Syntax) : List Name :=
  let stmt : TSyntax `tritonStmt := ⟨raw⟩
  match stmt with
  | `(tritonStmt| $id:ident = tl.make_block_ptr($_:tritonExpr, ($_:tritonExpr,),
      ($_:tritonExpr,), ($_:tritonExpr,), ($_:tritonExpr,), $_:ident=($_:tritonExpr,))) =>
      [id.getId.eraseMacroScopes]
  | _ => raw.getArgs.toList.flatMap pointerNames

def lower (body : Array (TSyntax `tritonStmt)) :
    TermElabM (Array (TSyntax `tritonStmt)) := do
  let allPointers := body.toList.flatMap (fun st => pointerNames st.raw)
  if allPointers.isEmpty then return body
  let used := body.toList.flatMap (fun st => names st.raw)
  return (← lowerBlock used allPointers body.toList [] 0).1.toArray

end VeriTile.Triton.DSL.RankOneBlockPtr
