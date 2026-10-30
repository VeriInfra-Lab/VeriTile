import VeriTile.Triton.DSL.Syntax
import Lean

/-! Source text frontend for the body of a Python Triton function. The stored
quotation remains unmodified. Comments/docstrings have no kernel instruction;
indentation becomes the existing structured statement syntax. -/
namespace VeriTile.Triton.DSL.PythonBody
open Lean Parser

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

private partial def takeBlock (input : List Char) (depth : Nat) (acc : List Char) : Except String String := do
  match input with
  | [] => throw "unterminated Triton function body"
  | c :: tail =>
      if c == '"' || c == '\'' then
        let (token, rest, _) ← stringToken c tail
        takeBlock rest depth (token.reverse ++ acc)
      else if c == '#' then
        let comment := tail.takeWhile (· != '\n')
        takeBlock (tail.drop comment.length) depth (comment.reverse ++ c :: acc)
      else if c == '{' then takeBlock tail (depth+1) (c :: acc)
      else if c == '}' then
        if depth == 1 then return String.ofList (c :: acc).reverse
        else takeBlock tail (depth-1) (c :: acc)
      else takeBlock tail depth (c :: acc)

private def blockFn : ParserFn := fun context state =>
  let text := context.extract state.pos context.inputString.rawEndPos
  match text.toList with
  | '{' :: tail =>
      match takeBlock tail 1 ['{'] with
      | .ok value => state.setPos (state.pos + value)
      | .error message => state.mkUnexpectedError message
  | _ => state.mkUnexpectedError "expected '{' before Python function body"

def sourceBlock : Parser := {fn := rawFn blockFn (trailingWs := Bool.true)}
@[combinator_parenthesizer sourceBlock]
def sourceBlock.parenthesizer : PrettyPrinter.Parenthesizer := PrettyPrinter.Parenthesizer.visitToken
@[combinator_formatter sourceBlock]
def sourceBlock.formatter : PrettyPrinter.Formatter := PrettyPrinter.Formatter.visitAtom Name.anonymous
declare_syntax_cat tritonPythonBody
syntax sourceBlock : tritonPythonBody

private structure Line where
  indent : Nat
  text : String
  deriving Inhabited

private def emit (lines : Array Line) (indent : Nat) (text : String) : Array Line :=
  let text := text.trimAscii.toString
  if text.isEmpty then lines else lines.push ⟨indent,text⟩

private partial def logicalLines (input : List Char) (lines : Array Line := #[])
    (indent : Nat := 0) (text : String := "") (delimiters : List (Char × Bool × Bool × Bool) := [])
    (atStart : Bool := Bool.true) : Except String (Array Line) := do
  match input with
  | [] =>
      unless delimiters.isEmpty do throw "unclosed delimiter in Python body"
      return emit lines indent text
  | c :: tail =>
      if c == '\t' && atStart then throw "tabs in Python indentation are unsupported"
      if atStart && c == ' ' then logicalLines tail lines (indent+1) text delimiters Bool.true
      else if c == '#' then
        logicalLines (tail.dropWhile (· != '\n')) lines indent text delimiters Bool.false
      else if c == '"' || c == '\'' then
        let (_, rest, value) ← stringToken c tail
        logicalLines rest lines indent (text ++ (Json.str value).compress) delimiters Bool.false
      else if c == '\n' || c == '\r' then
        if delimiters.isEmpty then logicalLines tail (emit lines indent text)
        else logicalLines tail lines indent (text ++ " ") delimiters Bool.false
      else if c == '\\' then
        match tail with
        | '\n' :: rest => logicalLines rest lines indent (text ++ " ") delimiters Bool.false
        | _ => throw "unsupported backslash outside Python string"
      else if c == '.' then
        let before := text.toList.reverse
        let digits := before.takeWhile (fun ch => ch.isDigit || ch == '_')
        let lead := before.drop digits.length
        let identifierBefore := lead.head?.any (fun ch => ch.isAlphanum || ch == '_' || ch == '.')
        if !digits.isEmpty && !identifierBefore && !tail.head?.any Char.isDigit then
          logicalLines tail lines indent (text ++ ".0") delimiters Bool.false
        else if digits.isEmpty && tail.head?.any Char.isDigit &&
            !before.head?.any (fun ch => ch.isAlphanum || ch == '_' || ch == '.') then
          logicalLines tail lines indent (text ++ "0.") delimiters Bool.false
        else logicalLines tail lines indent (text.push c) delimiters Bool.false
      else if c == '-' && tail.head? == some '-' then
        -- Prevent Lean's line-comment lexer from swallowing Python x--y.
        logicalLines tail.tail! lines indent (text ++ "- -") delimiters Bool.false
      else if ['(','[','{'].contains c then
        let preceding := text.trimAscii.toString
        let call := c == '(' && preceding.back?.any (fun ch => ch.isAlphanum || ['_',')',']'].contains ch)
        -- Native tuple/list shapes name the same static dimensions. Keep the
        -- raw Python tuple intact and adapt only this constructor argument
        -- to the existing dimension-list parser.
        let compact := String.ofList (preceding.toList.filter (! ·.isWhitespace))
        let shapeTuple := c == '(' &&
          ["tl.zeros(","tl.full("].any (fun word : String => compact.endsWith word)
        -- The legacy grammar reserves range(0, as one token followed by
        -- antiquotes. Parentheses around this integer are a source-spelling
        -- adaptation that lets an unmodified Python call use the general rule.
        let word := String.ofList ((preceding.toList.reverse.takeWhile
          (fun ch => ch.isAlphanum || ch == '_')).reverse)
        let afterOpen := tail.dropWhile Char.isWhitespace
        if c == '(' && word == "range" && afterOpen.head? == some '0' then
          let afterZero := afterOpen.tail!.dropWhile Char.isWhitespace
          if afterZero.head? == some ',' then
            return ← logicalLines afterZero lines indent (text ++ "((0)")
              ((c,call,Bool.false,Bool.false) :: delimiters) Bool.false
        logicalLines tail lines indent (text.push (if shapeTuple then '[' else c))
          ((c,call,shapeTuple,Bool.false) :: delimiters) Bool.false
      else if c == ',' && delimiters.head?.any (fun frame => frame.2.2.1) then
        let frame := delimiters.head!
        logicalLines tail lines indent (text.push c)
          ((frame.1,frame.2.1,frame.2.2.1,Bool.true) :: delimiters.tail!) Bool.false
      else if [')',']','}'].contains c then
        let expected := if c == ')' then '(' else if c == ']' then '[' else '{'
        unless delimiters.head?.map Prod.fst == some expected do throw "unbalanced Python delimiter"
        let preceding := text.trimAscii.toString
        let frame := delimiters.head!
        if frame.2.2.1 && !frame.2.2.2 && !preceding.endsWith "[" then
          throw "constructor shape requires a tuple or list"
        let text := if c == ')' && (frame.2.1 || frame.2.2.1) && preceding.endsWith "," then
            String.ofList preceding.toList.dropLast
          else text
        logicalLines tail lines indent (text.push (if frame.2.2.1 then ']' else c)) delimiters.tail! Bool.false
      else if c == ';' && delimiters.isEmpty then
        logicalLines tail (emit lines indent text) indent "" [] Bool.false
      else logicalLines tail lines indent (text.push c) delimiters Bool.false

private def docstring (text : String) : Bool :=
  match Json.parse text with
  | .ok (.str _) => Bool.true
  | _ => Bool.false

/-- Mechanical source spelling adaptation only. Unsupported Python constructs
are rejected by the statement/expression compiler, never supplied a proof. -/
def render (raw : String) : Except String String := do
  let chars := raw.toList
  unless chars.head? == some '{' && chars.getLast? == some '}' do throw "invalid source body delimiters"
  let lines ← logicalLines chars.tail!.dropLast
  if lines.isEmpty then return ""
  let base := lines[0]!.indent
  let mut levels := [base]
  let mut pending := Bool.false
  let mut output : Array String := #[]
  for line in lines do
    if pending then
      unless line.indent > levels.head! do throw "expected indented Python block"
      levels := line.indent :: levels
      pending := Bool.false
    else
      while line.indent < levels.head! && levels.length > 1 do
        output := output.push "}"
        levels := levels.tail!
      unless line.indent == levels.head! do throw "inconsistent Python indentation"
    if docstring line.text then continue
    if line.text.endsWith ":" then
      unless ["if ","elif ","else:","for ","while "].any (fun word : String => line.text.startsWith word) do
        throw "unsupported Python block header"
      output := output.push (String.ofList line.text.toList.dropLast ++ " {")
      pending := Bool.true
    else output := output.push line.text
  if pending then throw "expected Python block body"
  for _ in levels.tail! do output := output.push "}"
  return "\n".intercalate output.toList

/-- The original raw token is preserved in the syntax node. -/
def rawText (body : TSyntax `tritonPythonBody) : String := body.raw[0].getAtomVal

def statements (environment : Environment) (body : TSyntax `tritonPythonBody) : MacroM (Array (TSyntax `tritonStmt)) := do
  let text ← match render (rawText body) with
    | .ok text => pure text
    | .error message => Macro.throwErrorAt body message
  let stx ← match Parser.runParserCategory environment `term ("triton {\n" ++ text ++ "\n}") with
    | .ok stx => pure stx
    | .error message => Macro.throwErrorAt body ("unsupported Python Triton body: " ++ message)
  let `(triton { $stmts:tritonStmt* }) := stx
    | Macro.throwErrorAt body "invalid compiled Python body"
  return stmts

end VeriTile.Triton.DSL.PythonBody
