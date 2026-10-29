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

/-- Python permits parentheses around a tuple assignment RHS. Normalize only
this spelling to the existing simultaneous-assignment compiler; the raw source
and operand/target order are retained. -/
syntax (name := tritonPythonParenthesizedAssignment)
  ident ", " ident (", " ident)* " = " "(" tritonExpr ", " tritonExpr (", " tritonExpr)* ")" : tritonStmt

/-- Positional block-pointer arguments have the same order as Triton's named
arguments. Normalize their spelling before dtype and pointer analysis; the
stored Python body and every argument expression remain intact. -/
syntax (name := tritonPythonPositionalBlockPtr)
  "tl.make_block_ptr(" tritonExpr "," "(" tritonExpr,* ")"
  "," "(" tritonExpr,* ")" "," "(" tritonExpr,* ")"
  "," "(" tritonExpr,* ")" "," "(" num,* ")" ")" : tritonExpr

private structure Line where
  indent : Nat
  text : String
  deriving Inhabited

/-- Detect only an unparenthesized conditional expression, outside strings
and delimiter groups. This is a source-spelling boundary, not an evaluation. -/
private partial def topLevelConditional (input : List Char) (depth : Nat := 0) : Bool :=
  match input with
  | [] => Bool.false
  | c :: tail =>
    if c == '\"' || c == '\'' then
      match stringToken c tail with
      | .ok (_, rest, _) => topLevelConditional rest depth
      | .error _ => Bool.false
    else if ['(', '[', '{'].contains c then topLevelConditional tail (depth + 1)
    else if [')', ']', '}'].contains c then topLevelConditional tail (depth - 1)
    else if c.isAlpha || c == '_' then
      let restOfWord := tail.takeWhile (fun ch => ch.isAlphanum || ch == '_')
      let word := String.ofList (c :: restOfWord)
      (depth == 0 && word == "if") || topLevelConditional (tail.drop restOfWord.length) depth
    else topLevelConditional tail depth

private def emit (lines : Array Line) (indent : Nat) (text : String) : Array Line :=
  let text := text.trimAscii.toString
  -- The parsed view puts a conditional assignment RHS inside parentheses.
  -- Python's logical line already supplies that expression boundary. Quoted
  -- strings, nested calls, comparisons and augmented assignments are untouched.
  let pieces := text.splitOn "="
  let lhs := pieces.head!.trimAscii.toString
  let rhs := (String.intercalate "=" pieces.tail).trimAscii.toString
  let ordinary := lhs.toList.head?.any (fun c => c.isAlpha || c == '_') &&
    lhs.toList.all (fun c => c.isAlphanum || c == '_')
  let wrap := lhs.endsWith "tl.constexpr" || (ordinary && topLevelConditional rhs.toList)
  let text := if !pieces.tail.isEmpty && wrap then lhs ++ " = (" ++ rhs ++ ")" else text
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
        if ["if ", "elif ", "else:", "for ", "while "].any
            (fun word : String => text.trimAscii.toString.startsWith word) then
          throw "multiple statements in an inline Python suite are unsupported"
        logicalLines tail (emit lines indent text) indent "" [] Bool.false
      else logicalLines tail lines indent (text.push c) delimiters Bool.false

/-- A one-line Python suite keeps its statement inside the header's block.
Ignore colons inside strings, slices and other delimited expressions. -/
private partial def inlineHeader (input : List Char) (depth : Nat := 0)
    (header : String := "") : Except String (Option (String × String)) := do
  match input with
  | [] => return none
  | c :: tail =>
    if c == '\"' || c == '\'' then
      let (token, rest, _) ← stringToken c tail
      inlineHeader rest depth (header ++ String.ofList token)
    else if ['(', '[', '{'].contains c then
      inlineHeader tail (depth + 1) (header.push c)
    else if [')', ']', '}'].contains c then
      inlineHeader tail (depth - 1) (header.push c)
    else if c == ':' && depth == 0 then
      let body := (String.ofList tail).trimAscii.toString
      if body.isEmpty then return none
      return some (header, body)
    else inlineHeader tail depth (header.push c)

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
    else if ["if ", "elif ", "else:", "for ", "while "].any
        (fun word : String => line.text.startsWith word) then
      if let some (header, body) ← inlineHeader line.text.toList then
        output := output.push (header ++ " {\n" ++ body ++ "\n}")
      else output := output.push line.text
    else output := output.push line.text
  if pending then throw "expected Python block body"
  for _ in levels.tail! do output := output.push "}"
  return "\n".intercalate output.toList

/-- These are ordinary Python identifiers but reserved by Lean's lexer.
Escape them only in the parsed view; raw source and quoted strings stay intact.
The underscore remains the actual register name, including when it is read. -/
private partial def escapeSourceIdentifiers (input : List Char) (output : String := "") :
    Except String String := do
  match input with
  | [] => return output
  | c :: tail =>
      if c == '\"' || c == '\'' then
        let (raw, rest, _) ← stringToken c tail
        escapeSourceIdentifiers rest (output ++ String.ofList raw)
      else if c.isAlpha || c == '_' then
        let restOfWord := tail.takeWhile (fun ch => ch.isAlphanum || ch == '_')
        let word := String.ofList (c :: restOfWord)
        let word := if ["do", "_", "partial", "prefix"].contains word then "«" ++ word ++ "»" else word
        escapeSourceIdentifiers (tail.drop restOfWord.length) (output ++ word)
      else escapeSourceIdentifiers tail (output.push c)

/-- The original raw token is preserved in the syntax node. -/
def rawText (body : TSyntax `tritonPythonBody) : String := body.raw[0].getAtomVal

/-- Lean lexes `value.to` as one dotted identifier, so an attribute cast can
parse as an ordinary helper call. Recover only the exact Python `.to` spelling
with a supported type attribute; other helper names/arguments remain unchanged.
This changes parsed syntax only, before call-site dtype tracking. -/
private partial def normalizeAttributeCasts (raw : Syntax) : MacroM Syntax := do
  let children ← raw.getArgs.mapM normalizeAttributeCasts
  let rebuilt := match raw with
    | .node info kind _ => Syntax.node info kind children
    | _ => raw
  if let `(tritonExpr| tl.make_block_ptr($base:tritonExpr,
      ($shape:tritonExpr,*), ($strides:tritonExpr,*), ($offsets:tritonExpr,*),
      ($block:tritonExpr,*), ($order:num,*))) := rebuilt then
    let baseKey := mkIdent `base
    let shapeKey := mkIdent `shape
    let stridesKey := mkIdent `strides
    let offsetsKey := mkIdent `offsets
    let blockKey := mkIdent `block_shape
    let orderKey := mkIdent `order
    return (← `(tritonExpr| tl.make_block_ptr($baseKey:ident=$base,
      $shapeKey:ident=($shape,*), $stridesKey:ident=($strides,*),
      $offsetsKey:ident=($offsets,*), $blockKey:ident=($block,*),
      $orderKey:ident=($order,*)))).raw
  if let `(tritonStmt| $a:ident, $b:ident $[, $ids:ident]* =
      ($x:tritonExpr, $y:tritonExpr $[, $values:tritonExpr]*)) := rebuilt then
    return (← `(tritonStmt| $a:ident, $b:ident $[, $ids:ident]* =
      $x:tritonExpr, $y:tritonExpr $[, $values:tritonExpr]*)).raw
  unless rebuilt.getKind == ``tritonSourceHelperCall do return rebuilt
  let some callee := children[0]? | return rebuilt
  unless callee.isIdent do return rebuilt
  let .str receiver "to" := callee.getId.eraseMacroScopes | return rebuilt
  let some argument := children[2]? | return rebuilt
  let `(tritonExpr| $target:ident) := argument | return rebuilt
  let supported := match target.getId.eraseMacroScopes with
    | .str (.str _ "dtype") "element_ty"
    | .str (.str _ "type") "element_ty"
    | .str _ "dtype" => Bool.true
    | _ => Bool.false
  unless supported do return rebuilt
  let receiver := mkIdentFrom callee receiver
  let expression ← `(tritonExpr| $receiver:ident)
  let targetTerm ← `(term| $target:ident)
  return (← `(tritonExpr| ($expression).to($targetTerm:term))).raw

def statements (environment : Environment) (body : TSyntax `tritonPythonBody) : MacroM (Array (TSyntax `tritonStmt)) := do
  let text ← match render (rawText body) with
    | .ok text => pure text
    | .error message => Macro.throwErrorAt body message
  let text ← match escapeSourceIdentifiers text.toList with
    | .ok text => pure text
    | .error message => Macro.throwErrorAt body message
  let stx ← match Parser.runParserCategory environment `term ("triton {\n" ++ text ++ "\n}") with
    | .ok stx => pure stx
    | .error message => Macro.throwErrorAt body ("unsupported Python Triton body: " ++ message)
  let `(triton { $stmts:tritonStmt* }) := stx
    | Macro.throwErrorAt body "invalid compiled Python body"
  let stmts ← stmts.mapM fun st => do
    return (⟨← normalizeAttributeCasts st.raw⟩ : TSyntax `tritonStmt)
  -- A no-value return at the end of the function is its ordinary exit.
  -- Normalize only this top-level terminal statement. Nested returns and
  -- nonterminal returns still require the existing control-flow semantics;
  -- they must never become a loop-local no-op.
  if let some last := stmts.back? then
    if let `(tritonStmt| return) := last then return stmts.pop
  return stmts

private partial def numericLiteral (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $_:num) | `(tritonExpr| $_:scientific) => Bool.true
  | `(tritonExpr| ($inner:tritonExpr)) | `(tritonExpr| -$inner:tritonExpr) => numericLiteral inner
  | _ => Bool.false

/-- Mark weak numeric operands only in copied source comparisons/selects.
The legacy DSL keeps its original Real literal convention. This marker is
compiler metadata, not another source statement or floating operation. -/
partial def markWeakLiterals (raw : Syntax) : MacroM Syntax := do
  let args ← raw.getArgs.mapM markWeakLiterals
  let rebuilt := match raw with
    | .node info kind _ => Syntax.node info kind args
    | _ => raw
  let mark (value : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
    if numericLiteral value then `(tritonExpr| tl.__python_literal($value:tritonExpr))
    else pure value
  let expression : TSyntax `tritonExpr := ⟨rebuilt⟩
  match expression with
  | `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, $no:tritonExpr)) =>
      let yes ← mark yes
      let no ← mark no
      return (← `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, $no:tritonExpr))).raw
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr < $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr <= $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr > $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr >= $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr == $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) =>
      let a ← mark a
      let b ← mark b
      return (← `(tritonExpr| $a:tritonExpr != $b:tritonExpr)).raw
  | _ => return rebuilt

end VeriTile.Triton.DSL.PythonBody
