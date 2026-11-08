import VeriTile.Triton.DSL.ControlSource
import VeriTile.Triton.DSL.PythonShapeTuples

/-! Paste a complete Python function body into the shared control-source AST.
Only indentation, comments and shape punctuation are normalized. No branch,
cast, loop or source operand is specialized by this frontend. -/
namespace VeriTile.Triton.DSL.PastedControlSource
open Lean Elab Term Parser
def inlineOnly : Parser where
  info := epsilonInfo
  fn := fun _ state => if checkTailLinebreak state.stxStack.back then
    state.mkError "conditional expression must continue on the same line" else state
@[combinator_formatter inlineOnly]
def inlineOnlyFormatter : PrettyPrinter.Formatter := pure ()
@[combinator_parenthesizer inlineOnly]
def inlineOnlyParenthesizer : PrettyPrinter.Parenthesizer := pure ()
attribute [run_parser_attribute_hooks] inlineOnly
syntax:10 (name := conditionalExpr) tritonExpr:11 inlineOnly " if " tritonExpr:11 " else " tritonExpr:10 : tritonExpr
-- Raw Python booleans normalize to the ordinary boolean token spelling.
syntax:max tritonExpr:max noWs "." "to" "(" tritonDType "," "bitcast=true" ")" : tritonExpr
syntax:max ident noWs "." "to" "(" tritonDType "," "bitcast=true" ")" : tritonExpr
syntax ident "." "to" "(" tritonDType "," "bitcast=true" ")" : tritonExpr
syntax:max ident noWs "(" tritonDType "," "bitcast=true" ")" : tritonExpr

structure Invocation where
  signature : CallSignature
  constexprBools : List (String × Bool)
  body : VeriTile.Triton.ControlSource.Program

declare_syntax_cat controlBoolBinding
syntax ident " := " term : controlBoolBinding
syntax (name := pastedSource) "with_control_call " term " constexpr_bools "
  "[" controlBoolBinding,* "]" " in " "triton " RawSourceText.body : term
@[term_elab pastedSource]
def elaborateSource : TermElab := fun stx expected => do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source:=PythonCallSyntax.normalize (PythonShapeTuples.normalize source)
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("source_program triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed | .error message => throwErrorAt raw "{message}\n{source}"
  let signature : TSyntax `term := ⟨stx[1]⟩
  let bindings := stx[4].getSepArgs
  let values ← bindings.mapM fun binding => do
    let name:=quote binding[0].getId.eraseMacroScopes.toString
    let value : TSyntax `term := ⟨binding[2]⟩
    `(($name, $value))
  let body : TSyntax `term := ⟨parsed⟩
  let result ← `(({signature := $signature, constexprBools := [$values,*],body := $body} : Invocation))
  elabTerm result expected
end VeriTile.Triton.DSL.PastedControlSource
