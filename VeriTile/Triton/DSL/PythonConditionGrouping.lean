import VeriTile.Triton.DSL.SingleAssignmentPointerOffsets

/-! Parentheses required only by the internal expression grammar. Python source
keeps its original Boolean chains and assignment conditionals. Grouping retains
operand order and every operation; no global expression grammar is changed. -/
namespace VeriTile.Triton.DSL.PythonConditionGrouping
open Lean Elab Term

/-- Split a top-level Python word, ignoring strings and nested expressions. -/
def splitWord (source word : String) : List String := Id.run do
  let chars:=source.toList.toArray
  let token:=word.toList.toArray
  let mut quote : Option Char:=none
  let mut escaped:=Bool.false
  let mut depth : Int:=0
  let mut start:=0
  let mut i:=0
  let mut parts:=[]
  while i<chars.size do
    let c:=chars[i]!
    if let some q:=quote then
      if escaped then escaped:=Bool.false
      else if c=='\\' then escaped:=Bool.true
      else if c==q then quote:=none
    else if c=='"' || c=='\'' then quote:=some c
    else if ['(','[','{'].contains c then depth:=depth+1
    else if [')',']','}'].contains c then depth:=depth-1
    else if depth==0 && (chars.extract i (i+token.size))==token &&
        !(chars[i-1]?.any (fun c=>i>0 && (c.isAlphanum || c=='_'))) &&
        !(chars[i+token.size]?.any (fun c=>c.isAlphanum || c=='_')) then
      parts:=parts++[String.ofList ((chars.extract start i).toList)]
      i:=i+token.size-1
      start:=i+1
    i:=i+1
  return parts++[String.ofList ((chars.extract start chars.size).toList)]

def groupAnd (condition : String) : Except String String := do
  unless (splitWord condition "or").length==1 do
    throw "Python condition grouping does not yet support top-level or"
  let parts:=splitWord condition "and"
  if parts.length≤2 then return condition
  let first::rest:=parts | throw "empty Boolean condition"
  unless parts.all (fun s=> !s.trimAscii.toString.isEmpty) do throw "empty Boolean operand"
  return rest.foldl (fun left right=>"("++left++" and "++right++")") first

private def line (source : String) : Except String String := do
  let source:=source.trimAscii.toString
  if source.startsWith "if " && source.endsWith "{" then
    let condition:=((source.drop 3).dropEnd 1).toString
    return "if "++(← groupAnd condition)++" {"
  let [name,value]:=source.splitOn " = " | return source
  let conditional:=splitWord value "if"
  if conditional.length==1 then return source
  unless conditional.length==2 && (splitWord value "else").length==2 do
    throw "Python assignment grouping requires one complete conditional expression"
  return name++" = ("++value++")"

def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let lines ← (source.splitOn "\n").mapM fun s=>match line s with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" lines++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab PointerArguments.withCallOffsets]
def elaborateInvocation : TermElab := fun stx expected => do
  SingleAssignmentPointerOffsets.elaborateInvocation (← prepareSource stx) expected
@[term_elab PointerArguments.withSignatureOffsets]
def elaborateSignature : TermElab := fun stx expected => do
  SingleAssignmentPointerOffsets.elaborateSignature (← prepareSource stx) expected
end VeriTile.Triton.DSL.PythonConditionGrouping
