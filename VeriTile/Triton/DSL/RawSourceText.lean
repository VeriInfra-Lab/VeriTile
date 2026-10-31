import Lean
namespace VeriTile.Triton.DSL.RawSourceText
open Lean Parser

private partial def scanTriple (delimiter : Char) (escaped : Bool) : ParserFn := fun context state =>
  if context.atEnd state.pos then state.mkUnexpectedError "unterminated Python triple-quoted string"
  else
    let ch := context.get state.pos
    let next := context.next state.pos
    let after := context.next next
    if !escaped && ch==delimiter && context.get next==delimiter && context.get after==delimiter then
      state.setPos (context.next after)
    else scanTriple delimiter (!escaped && ch=='\\') context (state.setPos next)

private partial def scan (depth : Nat) (quote : Option Char) (escaped lineComment : Bool)
    (blockComment : Nat) : ParserFn := fun context state => Id.run do
  let pos := state.pos
  if context.atEnd pos then return state.mkUnexpectedError "unterminated Triton source body"
  let ch := context.get pos
  let nextPos := context.next pos
  let next := context.get nextPos
  let advance := state.setPos nextPos
  if lineComment then return scan depth none false (ch != '\n') 0 context advance
  if 0<blockComment then
    if ch=='/' && next=='-' then
      return scan depth none false false (blockComment+1) context (state.setPos (context.next nextPos))
    if ch=='-' && next=='/' then
      return scan depth none false false (blockComment-1) context (state.setPos (context.next nextPos))
    return scan depth none false false blockComment context advance
  if let some delimiter := quote then
    if escaped then return scan depth quote false false 0 context advance
    if ch=='\\' then return scan depth quote true false 0 context advance
    return scan depth (if ch==delimiter then none else quote) false false 0 context advance
  if ch=='#' || (ch=='-' && next=='-') then return scan depth none false true 0 context advance
  if ch=='/' && next=='-' then return scan depth none false false 1 context (state.setPos (context.next nextPos))
  if ch=='"' || ch=='\'' then
    let third := context.next nextPos
    if next==ch && context.get third==ch then
      let after := scanTriple ch false context (state.setPos (context.next third))
      if after.hasError then return after
      return scan depth none false false 0 context after
    return scan depth (some ch) false false 0 context advance
  if ch=='{' then return scan (depth+1) none false false 0 context advance
  if ch=='}' then
    if depth==1 then return advance
    else return scan (depth-1) none false false 0 context advance
  return scan depth none false false 0 context advance

def body : Parser := {fn:=rawFn (fun context state =>
  if context.get state.pos != '{' then state.mkUnexpectedError "expected '{' before Triton body"
  else scan 1 none false false 0 context (state.next context state.pos)) (trailingWs:=true)}

@[combinator_formatter body] def bodyFormatter : PrettyPrinter.Formatter :=
  PrettyPrinter.Formatter.visitAtom Name.anonymous
@[combinator_parenthesizer body] def bodyParenthesizer : PrettyPrinter.Parenthesizer :=
  PrettyPrinter.Parenthesizer.visitToken
attribute [run_parser_attribute_hooks] body

structure Line where
  indent : Nat
  text : String
  deriving Inhabited

/-- These valid Python names are reserved tokens in Lean. Escaping changes
only parser spelling; the resulting identifier still has its original name. -/
private def cleanWord (word : String) : String :=
  if word == "True" then "true" else if word == "False" then "false"
  else if word == "_" || word == "do" then "«" ++ word ++ "»" else word

/-- Token spelling only: Python comments and booleans, single-quoted strings,
and trailing decimal points. No arithmetic or instruction is simplified. -/
private def cleanLine (line : String) : Except String (String × Int) := do
  let chars := line.toList.toArray
  let mut out := ""
  let mut word := ""
  let mut quote : Option Char := none
  let mut nesting : Int := 0
  let mut i := 0
  while i<chars.size do
    let ch := chars[i]!
    let next := chars[i+1]?
    if let some delimiter := quote then
      if ch=='\\' then
        let some escaped := next | throw "unfinished Python string escape"
        unless ['n','r','t','\\','\'','"','0'].contains escaped do throw "unsupported Python string escape"
        if escaped=='\'' then out:=out.push '\''
        else out:=out.push '\\' |>.push escaped
        i:=i+2
      else if ch==delimiter then
        out:=out.push '"'
        quote:=none
        i:=i+1
      else
        if ch=='"' then out:=out++"\\\"" else out:=out.push ch
        i:=i+1
    else if ch.isAlphanum || ch=='_' then
      word:=word.push ch
      i:=i+1
    else
      let token := cleanWord word
      out:=out++token
      let numeric := !word.isEmpty && word.toList.all Char.isDigit
      word:=""
      if ch=='#' then break
      if ch=='"' || ch=='\'' then
        quote:=some ch
        out:=out.push '"'
      else
        out:=out.push ch
        if ch=='-' && next==some '-' then out:=out.push ' '
        if ch=='.' && numeric && !(next.any Char.isDigit) then out:=out.push '0'
        if ['(','[','{'].contains ch then nesting:=nesting+1
        if [')',']','}'].contains ch then nesting:=nesting-1
      i:=i+1
  if quote.isSome then throw "multiline strings are unsupported in Triton bodies"
  let token := cleanWord word
  return (out++token,nesting)

private partial def quoted (q : Char) (triple : Bool) (input : List Char)
    (raw value : List Char := []) : Except String (List Char × List Char × String) := do
  match input with
  | [] => throw "unterminated Python string"
  | a :: b :: c :: tail =>
      if triple && a == q && b == q && c == q then
        return ((c :: b :: a :: raw).reverse, tail, String.ofList value.reverse)
      else consume a (b :: c :: tail) raw value
  | a :: tail =>
      if !triple && a == q then return ((a :: raw).reverse, tail, String.ofList value.reverse)
      else consume a tail raw value
where
  consume (a : Char) (tail raw value : List Char) : Except String (List Char × List Char × String) := do
    if !triple && a == q then return ((a :: raw).reverse, tail, String.ofList value.reverse)
    if a == '\\' then
      match tail with
      | [] => throw "unterminated Python escape"
      | b :: rest =>
          if b == '\n' then return ← quoted q triple rest (b :: a :: raw) value
          let decoded ← match b with
            | 'n' => pure '\n' | 'r' => pure '\r' | 't' => pure '\t'
            | '\\' => pure '\\' | '\'' => pure '\'' | '"' => pure '"'
            | _ => throw "unsupported Python string escape"
          quoted q triple rest (b :: a :: raw) (decoded :: value)
    else quoted q triple tail (a :: raw) (a :: value)

private def stringToken (q : Char) (tail : List Char) : Except String (List Char × List Char × String) :=
  match tail with
  | a :: b :: rest =>
      if a == q && b == q then quoted q Bool.true rest [q,q,q] []
      else quoted q Bool.false tail [q] []
  | _ => quoted q Bool.false tail [q] []

private def emit (lines : Array Line) (indent : Nat) (text : String) : Array Line :=
  let text := text.trimAscii.toString
  if text.isEmpty then lines else lines.push ⟨indent,text⟩

private partial def pythonLogicalLines (input : List Char) (lines : Array Line := #[])
    (indent : Nat := 0) (text : String := "") (delimiters : List (Char × Bool) := [])
    (atStart : Bool := Bool.true) : Except String (Array Line) := do
  match input with
  | [] =>
      unless delimiters.isEmpty do throw "unclosed delimiter in Python body"
      return emit lines indent text
  | c :: tail =>
      if c == '\t' && atStart then throw "tabs in Python indentation are unsupported"
      if atStart && c == ' ' then pythonLogicalLines tail lines (indent+1) text delimiters Bool.true
      else if c == '#' then
        pythonLogicalLines (tail.dropWhile (· != '\n')) lines indent text delimiters Bool.false
      else if c == '"' || c == '\'' then
        let (_, rest, value) ← stringToken c tail
        pythonLogicalLines rest lines indent (text ++ (Json.str value).compress) delimiters Bool.false
      else if c == '\n' || c == '\r' then
        if delimiters.isEmpty then pythonLogicalLines tail (emit lines indent text)
        else pythonLogicalLines tail lines indent (text ++ " ") delimiters Bool.false
      else if c == '\\' then
        match tail with
        | '\n' :: rest => pythonLogicalLines rest lines indent (text ++ " ") delimiters Bool.false
        | _ => throw "unsupported backslash outside Python string"
      else if c == '-' && tail.head? == some '-' then
        -- Prevent Lean's line-comment lexer from swallowing Python x--y.
        pythonLogicalLines tail.tail! lines indent (text ++ "- -") delimiters Bool.false
      else if ['(','[','{'].contains c then
        let preceding := text.trimAscii.toString
        let call := c == '(' && preceding.back?.any (fun ch => ch.isAlphanum || ['_',')',']'].contains ch)
        pythonLogicalLines tail lines indent (text.push c) ((c,call) :: delimiters) Bool.false
      else if [')',']','}'].contains c then
        let expected := if c == ')' then '(' else if c == ']' then '[' else '{'
        unless delimiters.head?.map Prod.fst == some expected do throw "unbalanced Python delimiter"
        let preceding := text.trimAscii.toString
        let text := if c == ')' && delimiters.head!.2 && preceding.endsWith "," then
            String.ofList preceding.toList.dropLast
          else text
        pythonLogicalLines tail lines indent (text.push c) delimiters.tail! Bool.false
      else if c == ';' && delimiters.isEmpty then
        pythonLogicalLines tail (emit lines indent text) indent "" [] Bool.false
      else pythonLogicalLines tail lines indent (text.push c) delimiters Bool.false

private def docstring (text : String) : Bool :=
  match Json.parse text with
  | .ok (.str _) => Bool.true
  | _ => Bool.false

private def logicalLines (source : String) : Except String (Array Line) := do
  let lines ← pythonLogicalLines source.toList
  let mut normalized := #[]
  for line in lines do
    if docstring line.text then continue
    let (text,_) ← cleanLine line.text
    normalized := normalized.push {line with text := text}
  return normalized

mutual
  private partial def suite (lines : Array Line) (start indent : Nat) : Except String (String × Nat) := do
    let mut out := ""
    let mut index := start
    while index<lines.size do
      let line := lines[index]!
      if line.indent<indent then break
      if line.indent>indent then throw "unexpected indentation in Triton body"
      if line.text=="else:" || line.text.startsWith "elif " then break
      let (text,next) ← statement lines index false
      out:=out++text++"\n"
      index:=next
    return (out,index)

  private partial def statement (lines : Array Line) (index : Nat) (elifAllowed : Bool) : Except String (String × Nat) := do
    let line := lines[index]!
    let text := if elifAllowed && line.text.startsWith "elif " then "if "++(line.text.drop 5).toString else line.text
    if !text.endsWith ":" then return (text,index+1)
    unless text.startsWith "if " || text.startsWith "for " || text.startsWith "while " do
      throw "unsupported compound statement in Triton body"
    let some child := lines[index+1]? | throw "missing Triton suite"
    unless line.indent<child.indent do throw "expected an indented Triton suite"
    let (inner,next) ← suite lines (index+1) child.indent
    let mut out := (text.dropEnd 1).toString++" {\n"++inner++"}"
    let mut next := next
    if text.startsWith "if " then
      if let some branch := lines[next]? then
        if branch.indent==line.indent then
          if branch.text.startsWith "elif " then
            let (branch,after) ← statement lines next true
            out:=out++" else {\n"++branch++"\n}"
            next:=after
          else if branch.text=="else:" then
            let some child := lines[next+1]? | throw "missing else suite"
            unless branch.indent<child.indent do throw "expected an indented else suite"
            let (branch,after) ← suite lines (next+1) child.indent
            out:=out++" else {\n"++branch++"}"
            next:=after
    return (out,next)
end

/-- Preserve the source statements and use braces solely as the internal DSL's
suite delimiter. Existing brace syntax remains accepted for compatibility. -/
def normalize (raw : String) : Except String String := do
  let source := ((raw.drop 1).dropEnd 1).toString
  let startsSuite := fun (line : String) =>
    ["if ","for ","while ","elif ","else:"].any (fun introducer => line.startsWith introducer)
  let physical := (source.splitOn "\n").map (fun line => line.trimAscii.toString)
  let python := physical.any (fun line => startsSuite line && line.endsWith ":")
  if !python && physical.any (fun line => startsSuite line && line.contains '{') then return source
  let lines ← logicalLines source
  if lines.isEmpty then return ""
  if !lines.any (fun line => line.text.endsWith ":") then
    return String.intercalate "\n" (lines.toList.map (·.text))
  let (body,after) ← suite lines 0 lines[0]!.indent
  unless after==lines.size do throw "unmatched else/elif or inconsistent source indentation"
  return body
end VeriTile.Triton.DSL.RawSourceText
