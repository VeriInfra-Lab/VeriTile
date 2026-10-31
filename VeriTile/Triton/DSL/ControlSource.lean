import VeriTile.Triton.DSL.BoolTerms
import Lean

/-! Shared concrete source trees for control-flow kernels. Expressions retain
all parsed tokens and operands, while loops, branches and calls have explicit
nodes. This frontend selects no numerical model or concurrency semantics.
A typed backend must reject unsupported forms and prove its source connection.
-/
namespace VeriTile.Triton.ControlSource
open Lean

/-- Concrete expression syntax, with positions and macro scopes removed by
the quotation. Syntax kinds, tokens and every operand remain explicit data. -/
abbrev Expr := Syntax

def Expr.toSyntax (e : Expr) : Syntax := e

structure Argument where
  keyword : Option String := none
  value : Expr
  deriving Repr

/-- Shared source control flow. A call retains its complete argument list;
no task-specific helper is replaced by a whole-program callback. -/
inductive Statement where
  | assign (target : String) (value : Expr)
  | update (target operation : String) (value : Expr)
  | call (function : String) (arguments : List Argument)
  | bindCall (targets : List String) (function : String) (arguments : List Argument)
  | branch (condition : Expr) (yes no : List Statement)
  | range (index : String) (bounds : List Expr) (body : List Statement)
  | whileLoop (condition : Expr) (body : List Statement)
  | returnValues (values : List Expr)
  | pass
  deriving Repr

abbrev Program := List Statement

declare_syntax_cat controlSourceName
syntax ident : controlSourceName
syntax "end" : controlSourceName
declare_syntax_cat controlSourceStmt
syntax controlSourceName " = " tritonExpr : controlSourceStmt
syntax controlSourceName " += " tritonExpr : controlSourceStmt
syntax controlSourceName " -= " tritonExpr : controlSourceStmt
syntax ident "(" tritonExpr,* ")" : controlSourceStmt
syntax ident ", " ident " = " ident "(" tritonExpr,* ")" : controlSourceStmt
syntax "tl.static_assert(" tritonExpr ")" : controlSourceStmt
syntax "tl.debug_barrier()" : controlSourceStmt
syntax "tl.atomic_xchg(" tritonExpr ", " tritonExpr ")" : controlSourceStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ")" : controlSourceStmt
syntax "tl.store(" tritonExpr ", " tritonExpr "," tritonMemKwarg ")" : controlSourceStmt
syntax "tl.atomic_add(" tritonExpr ", " tritonExpr "," tritonMemKwarg ")" : controlSourceStmt
syntax "if " tritonExpr " { " controlSourceStmt* " }" : controlSourceStmt
syntax "if " tritonExpr " { " controlSourceStmt* " }" "else" "{" controlSourceStmt* " }" : controlSourceStmt
syntax "while " tritonExpr " { " controlSourceStmt* " }" : controlSourceStmt
syntax "for " ident " in " "range(" tritonExpr,* ")" "{" controlSourceStmt* " }" : controlSourceStmt
syntax "for " ident " in " "range(0, " tritonExpr ")" "{" controlSourceStmt* " }" : controlSourceStmt
syntax "return" : controlSourceStmt
syntax "return " tritonExpr,+ : controlSourceStmt
syntax "pass" : controlSourceStmt
syntax:40 tritonExpr:41 " or " tritonExpr:41 : tritonExpr
syntax "end" : tritonExpr
syntax "tl.full(" "(" ")" ", " tritonExpr ", " tritonDType ")" : tritonExpr
syntax "source_program " "triton" "{" controlSourceStmt* " }" : term

private partial def quoteExpr (s : Syntax) : MacroM (TSyntax `term) := do
  match s with
  | .missing => `(Syntax.missing)
  | .atom _ token =>
    let value := quote token
    `(Syntax.atom SourceInfo.none $value)
  | .ident _ spelling name _ =>
    let spelling := quote spelling.toString
    let name := quote name.eraseMacroScopes
    `(Syntax.ident SourceInfo.none (String.toRawSubstring $spelling) $name [])
  | .node _ kind children =>
    let children ← children.mapM quoteExpr
    let kind := quote kind
    `(Syntax.node SourceInfo.none $kind #[$children,*])

private def sourceName (s : TSyntax `controlSourceName) : String :=
  if s.raw[0].isIdent then s.raw[0].getId.eraseMacroScopes.toString else "end"

private def argument (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  let e ← quoteExpr e.raw
  `({ value := $e : Argument })

private def call (name : String) (args : Array (TSyntax `tritonExpr))
    (mask : Option (TSyntax `tritonExpr) := none) : MacroM (TSyntax `term) := do
  let mut args ← args.mapM argument
  if let some mask := mask then
    let mask ← quoteExpr mask.raw
    args := args.push (← `({ keyword := some "mask", value := $mask : Argument }))
  let name := quote name
  `(Statement.call $name [$args,*])

/-- Reuse the ordinary memory keyword grammar so importing this frontend
keeps masked loads and stores available to the ordinary Triton quotation. -/
private def maskValue (arg : TSyntax `tritonMemKwarg) : MacroM (TSyntax `tritonExpr) := do
  match arg with
  | `(tritonMemKwarg| $name:ident = $value:tritonExpr) =>
    unless name.getId.eraseMacroScopes.toString == "mask" do
      Macro.throwErrorAt arg "control source effects require mask=expression"
    pure value
  | _ => Macro.throwErrorAt arg "control source effects require mask=expression"

private partial def quoteStatement (s : TSyntax `controlSourceStmt) : MacroM (TSyntax `term) := do
  match s with
  | `(controlSourceStmt| $target:controlSourceName = $value:tritonExpr) =>
    let value ← quoteExpr value.raw
    let target := quote (sourceName target)
    `(Statement.assign $target $value)
  | `(controlSourceStmt| $target:controlSourceName += $value:tritonExpr) =>
    let value ← quoteExpr value.raw
    let target := quote (sourceName target)
    `(Statement.update $target "+" $value)
  | `(controlSourceStmt| $target:controlSourceName -= $value:tritonExpr) =>
    let value ← quoteExpr value.raw
    let target := quote (sourceName target)
    `(Statement.update $target "-" $value)
  | `(controlSourceStmt| $a:ident, $b:ident = $function:ident($args:tritonExpr,*)) =>
    let args ← args.getElems.mapM argument
    let a := quote a.getId.eraseMacroScopes.toString
    let b := quote b.getId.eraseMacroScopes.toString
    let function := quote function.getId.eraseMacroScopes.toString
    `(Statement.bindCall [$a, $b] $function [$args,*])
  | `(controlSourceStmt| $function:ident($args:tritonExpr,*)) =>
    call function.getId.eraseMacroScopes.toString args.getElems
  | `(controlSourceStmt| tl.static_assert($e:tritonExpr)) => call "tl.static_assert" #[e]
  | `(controlSourceStmt| tl.debug_barrier()) => call "tl.debug_barrier" #[]
  | `(controlSourceStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr)) => call "tl.atomic_xchg" #[p,v]
  | `(controlSourceStmt| tl.store($p:tritonExpr, $v:tritonExpr)) => call "tl.store" #[p,v]
  | `(controlSourceStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw:tritonMemKwarg)) => do
    call "tl.store" #[p,v] (some (← maskValue kw))
  | `(controlSourceStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr, $kw:tritonMemKwarg)) => do
    call "tl.atomic_add" #[p,v] (some (← maskValue kw))
  | `(controlSourceStmt| if $c:tritonExpr { $yes:controlSourceStmt* } else { $no:controlSourceStmt* }) =>
    let c ← quoteExpr c.raw
    let yes ← yes.mapM quoteStatement
    let no ← no.mapM quoteStatement
    `(Statement.branch $c [$yes,*] [$no,*])
  | `(controlSourceStmt| if $c:tritonExpr { $yes:controlSourceStmt* }) =>
    let c ← quoteExpr c.raw
    let yes ← yes.mapM quoteStatement
    `(Statement.branch $c [$yes,*] [])
  | `(controlSourceStmt| while $c:tritonExpr { $body:controlSourceStmt* }) =>
    let c ← quoteExpr c.raw
    let body ← body.mapM quoteStatement
    `(Statement.whileLoop $c [$body,*])
  | `(controlSourceStmt| for $i:ident in range(0, $stop:tritonExpr) { $body:controlSourceStmt* }) =>
    let zero ← `(tritonExpr| 0)
    let zero ← quoteExpr zero.raw
    let stop ← quoteExpr stop.raw
    let body ← body.mapM quoteStatement
    let i := quote i.getId.eraseMacroScopes.toString
    `(Statement.range $i [$zero, $stop] [$body,*])
  | `(controlSourceStmt| for $i:ident in range($bounds:tritonExpr,*) { $body:controlSourceStmt* }) =>
    let bounds := bounds.getElems
    unless bounds.size ≥ 1 && bounds.size ≤ 3 do
      Macro.throwErrorAt s "range requires one, two or three source bounds"
    let bounds ← bounds.mapM fun e => quoteExpr e.raw
    let body ← body.mapM quoteStatement
    let i := quote i.getId.eraseMacroScopes.toString
    `(Statement.range $i [$bounds,*] [$body,*])
  | `(controlSourceStmt| return) => `(Statement.returnValues [])
  | `(controlSourceStmt| return $values:tritonExpr,*) =>
    let values ← values.getElems.mapM fun e => quoteExpr e.raw
    `(Statement.returnValues [$values,*])
  | `(controlSourceStmt| pass) => `(Statement.pass)
  | _ => Macro.throwErrorAt s "unsupported control source statement"

macro_rules
  | `(source_program triton { $body:controlSourceStmt* }) => do
    let body ← body.mapM quoteStatement
    `(([$body,*] : Program))

end VeriTile.Triton.ControlSource
