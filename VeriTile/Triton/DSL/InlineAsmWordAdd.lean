import VeriTile.Triton.DSL.PythonFullConstructors

/-! The exact pure two-input PTX add.u32 ABI is a modulo-32 integer addition.
Only this instruction, constraints, dtype and packing are supported. The pasted
source retains the assembly call; other assembly is rejected, not interpreted
as this operation. -/
namespace VeriTile.Triton.DSL.InlineAsmWordAdd
open Lean Elab Term Meta

syntax (name := wordAddAsm) "tl.inline_asm_elementwise(" str ", "
  &"constraints" "=" str ", " &"args" "=" "[" tritonExpr,* "]" ", "
  &"dtype" "=" tritonDType ", " &"is_pure" "=" tritonExpr ", " &"pack" "=" num ")" : tritonExpr

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  if raw.getKind==``wordAddAsm then
    let `(tritonExpr| tl.inline_asm_elementwise($code:str, constraints=$constraints:str,
        args=[$arguments:tritonExpr,*], dtype=$dtype:tritonDType,
        is_pure=$pure:tritonExpr, pack=$pack:num)) := raw
      | Macro.throwError "expected a complete assembly ABI"
    let unsignedAdd := code.getString=="add.u32 $0, $1, $2;"
    let int32 := match dtype with | `(tritonDType| tl.int32) => Bool.true | _ => Bool.false
    let pureFlag := match pure with | `(tritonExpr| true) => Bool.true | _ => Bool.false
    unless unsignedAdd && constraints.getString=="=r,r,r" && int32 && pureFlag &&
        pack.getNat==1 && arguments.getElems.size==2 do
      Macro.throwError "unsupported inline assembly: expected pure add.u32 with =r,r,r, two inputs, int32 and pack=1"
    let a : TSyntax `tritonExpr := ⟨← lower arguments.getElems[0]!.raw⟩
    let b : TSyntax `tritonExpr := ⟨← lower arguments.getElems[1]!.raw⟩
    return (← `(tritonExpr| (($a:tritonExpr + $b:tritonExpr).to(tl.uint32)).to(tl.int32))).raw
  return raw.setArgs (← raw.getArgs.mapM lower)

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => liftMacroM (lower parsed) | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete assembly body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  CallSignature.elaborateInvocation (← prepare stx) expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  CallSignature.elaborateSignature (← prepare stx) expected
end VeriTile.Triton.DSL.InlineAsmWordAdd
