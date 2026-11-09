import VeriTile.Triton.DSL.PythonCallSyntax

/-! Single-line Python suites use the same typed control flow as braced
suites. Every condition and branch body is retained; caller constants are
specialized by the existing invocation elaborator. -/
namespace VeriTile.Triton.DSL.PythonInlineSuites
open Lean Elab Term

private def colon? (line : String) : Option Nat := Id.run do
  let chars := line.toList.toArray
  let mut quote : Option Char := none
  let mut escaped := Bool.false
  let mut depth := 0
  for i in [:chars.size] do
    let ch := chars[i]!
    if let some delimiter := quote then
      if escaped then escaped := Bool.false
      else if ch=='\\' then escaped := Bool.true
      else if ch==delimiter then quote := none
    else if ch=='"' || ch=='\'' then quote := some ch
    else if ['(','[','{'].contains ch then depth := depth+1
    else if [')',']','}'].contains ch then depth := depth-1
    else if ch==':' && depth==0 then return some i
  return none

/-- Apply only to suite headers. Colons in strings and subscripts are not
suite delimiters, and ordinary assignments and multiline suites are unchanged. -/
def normalize (source : String) : String :=
  if (source.splitOn "\"\"\"").length > 1 || (source.splitOn "'''").length > 1 then source else
  String.intercalate "\n" ((source.splitOn "\n").map fun line => Id.run do
    let text := line.trimAscii.toString
    unless text.startsWith "if " || text.startsWith "elif " || text.startsWith "else:" do
      return line
    let some pos := colon? line | return line
    let chars := line.toList.toArray
    let body := (String.ofList (chars.toList.drop (pos+1))).trimAscii.toString
    if body.isEmpty || body.startsWith "#" then return line
    let header := String.ofList (chars.toList.take pos)
    return header++" { "++body++" }")

/-- Escaping a Lean keyword preserves the original Python identifier name.
Strings, comments and already escaped identifiers are copied unchanged. -/
def escapeNames (source : String) : String := Id.run do
  if (source.splitOn "\"\"\"").length > 1 || (source.splitOn "'''").length > 1 then return source
  let mut out := ""
  let mut word := ""
  let mut quote : Option Char := none
  let mut escaped := Bool.false
  let mut named := Bool.false
  let mut comment := Bool.false
  for ch in source.toList do
    if comment then
      out := out.push ch
      if ch=='\n' then comment := Bool.false
    else if named then
      out := out.push ch
      if ch=='»' then named := Bool.false
    else if let some delimiter := quote then
      out := out.push ch
      if escaped then escaped := Bool.false
      else if ch=='\\' then escaped := Bool.true
      else if ch==delimiter then quote := none
    else if ch.isAlphanum || ch=='_' then word := word.push ch
    else
      out := out ++ (if word=="partial" then "«partial»" else word)
      word := ""
      out := out.push ch
      if ch=='"' || ch=='\'' then quote := some ch
      if ch=='«' then named := Bool.true
      if ch=='#' then comment := Bool.true
  return out ++ (if word=="partial" then "«partial»" else word)

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let normalized := normalize (escapeNames source)
  if normalized==source then return stx
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++normalized++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  PythonCallSyntax.elaborateSignature (← prepare stx) expected

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  PythonCallSyntax.elaborateInvocation (← prepare stx) expected

end VeriTile.Triton.DSL.PythonInlineSuites
