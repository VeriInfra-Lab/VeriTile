import VeriTile.Triton.DSL.PastedControlSource
import VeriTile.Triton.DSL.PythonFullConstructors

/-! Typed caller metadata for mutually linked pasted control functions.
The typed backend must check the dtype constants and link every named helper
to the corresponding source member before executing the complete bundle. -/
namespace VeriTile.Triton.DSL.PastedControlFunctions
open Lean Elab Term Meta PastedControlSource

structure Invocation extends PastedControlSource.Invocation where
  entrypoint : String
  dtypeConstants : List (String × SourceDType)
  helperBindings : List (String × String)

syntax (name := pastedFunction) "with_control_function " str term " constexpr_bools "
  "[" controlBoolBinding,* "]" " dtype_constants " "[" controlBoolBinding,* "]"
  " jit_calls " "[" controlBoolBinding,* "]" " in " "triton " RawSourceText.body : term

private def bindingTerms (bindings : Syntax) : TermElabM (TSyntax `term) := do
  let values ← bindings.getSepArgs.mapM fun binding => do
    let name:=quote binding[0].getId.eraseMacroScopes.toString
    let value : TSyntax `term := ⟨binding[2]⟩
    `(($name,$value))
  `([$values,*])

@[term_elab pastedFunction]
def elaborateFunction : TermElab := fun stx expected => do
  let parent:=stx.setArgs #[stx[0],stx[2],stx[3],stx[4],stx[5],stx[6],stx[15],stx[16],stx[17]]
  let base ← PastedControlSource.elaborateSource parent (some (mkConst ``PastedControlSource.Invocation))
  let dtypes ← bindingTerms stx[9]
  let helpers ← bindingTerms stx[13]
  let dtypeType ← elabType (← `(List (String × SourceDType)))
  let helperType ← elabType (← `(List (String × String)))
  let dtypes ← elabTermEnsuringType dtypes dtypeType
  let helpers ← elabTermEnsuringType helpers helperType
  let some entry := stx[1].isStrLit? | throwError "literal control function name required"
  let result ← mkAppM ``Invocation.mk #[base,mkStrLit entry,dtypes,helpers]
  ensureHasType expected result
end VeriTile.Triton.DSL.PastedControlFunctions
