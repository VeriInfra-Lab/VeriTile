import VeriTile.Triton.DSL.CallSignature

/-! Python permits trailing commas in calls, lists and shape tuples. This
optional invocation extension preserves those literal source spellings and
feeds the existing typed elaborator its equivalent delimiter syntax. Other
singleton tuples are left intact, so they cannot silently become scalars. -/
namespace VeriTile.Triton.DSL.PythonCallSyntax
open Lean Elab Term

private structure Frame where
  close : Char
  trailing : Bool
  call : String := ""
  deriving Inhabited

private def callee (out : Array Char) : String :=
  String.ofList ((out.toList.reverse.dropWhile Char.isWhitespace).takeWhile
    (fun ch => ch.isAlphanum || ch=='_' || ch=='.')).reverse

/-- Change punctuation only outside strings/comments. A shape tuple is
recognized only as the first argument of zeros/ones/full, whose existing DSL
parsers already interpret parenthesized dimensions as a shape. -/
def normalize (source : String) : String := Id.run do
  let chars := source.toList.toArray
  let mut out : Array Char := #[]
  let mut frames : Array Frame := #[]
  let mut quote : Option Char := none
  let mut triple := Bool.false
  let mut escaped := Bool.false
  let mut comment := Bool.false
  let mut i := 0
  while i < chars.size do
    let ch := chars[i]!
    if comment then
      out := out.push ch
      if ch=='\n' then comment := Bool.false
      i := i+1
      continue
    if let some delimiter := quote then
      out := out.push ch
      if escaped then escaped := Bool.false
      else if ch=='\\' then escaped := Bool.true
      else if ch==delimiter then
        if triple then
          if chars[i+1]?==some delimiter && chars[i+2]?==some delimiter then
            out := out.push delimiter |>.push delimiter
            i := i+2
            quote := none
            triple := Bool.false
        else quote := none
      i := i+1
      continue
    if ch=='#' then
      comment := Bool.true
    else if ch=='"' || ch=='\'' then
      quote := some ch
      if chars[i+1]?==some ch && chars[i+2]?==some ch then
        triple := Bool.true
        out := out.push ch |>.push ch
        i := i+2
    else if ch=='(' then
      let name := callee out
      let previous := (out.toList.reverse.dropWhile Char.isWhitespace).head?
      let callable := (!name.isEmpty && !["if","elif","while","for","return","not","and","or","else","in"].contains name) ||
        previous==some ')' || previous==some ']'
      let shape := previous==some '(' && frames.back?.any
        (fun frame => ["tl.zeros","tl.ones","tl.full"].contains frame.call)
      frames := frames.push ⟨')',callable || shape,if callable then name else ""⟩
    else if ch=='[' then
      let previous := (out.toList.reverse.dropWhile Char.isWhitespace).head?
      let indexing := previous.any (fun ch => ch.isAlphanum || ch=='_' || ch==')' || ch==']')
      frames := frames.push ⟨']',!indexing,""⟩
    else if ch=='{' then
      frames := frames.push ⟨'}',Bool.false,""⟩
    else if [')',']','}'].contains ch then
      if let some frame := frames.back? then
        if frame.close==ch then
          if frame.trailing then
            let trimmed := out.toList.reverse.dropWhile Char.isWhitespace
            if trimmed.head?==some ',' then out := trimmed.tail!.reverse.toArray
          frames := frames.pop
    out := out.push ch
    i := i+1
  return String.ofList out.toList

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let normalized := normalize source
  if normalized==source then return stx
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++normalized++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  CallSignature.elaborateSignature (← prepare stx) expected

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  CallSignature.elaborateInvocation (← prepare stx) expected

end VeriTile.Triton.DSL.PythonCallSyntax
