import VeriTile.Triton.DSL

open Lean Elab Term
open VeriTile.Triton VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast

/-! The source declares every storage dtype and every signed scalar cast.
The preliminary Nat-use hints would override explicit int64 assignments.
This adapter suppresses only those hints and uses the shared statement
expander, floating provenance and precision scoping. Shared `tl.cdiv` only
accepts Nat, so its ordinary `(a+b-1)//b` expansion is performed syntactically,
retaining the signed scalar operands. No evaluator is introduced. -/
syntax "llama_triton" "{" tritonStmt* "}" : term

private partial def signedCdiv (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args => pure (.node info kind (← args.mapM signedCdiv))
    | other => pure other
  let e : TSyntax `tritonExpr := ⟨raw⟩
  match e with
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
      return (← `(tritonExpr| (($a + $b - $((1 : Nat))) // $b))).raw
  | _ => return raw

elab_rules : term
  | `(llama_triton { $stmts:tritonStmt* }) => do
      let expanded ← liftMacroM do
        let original ← stmts.toList.mapM fun st => do
          return (⟨← signedCdiv (← Inference.normalizeControlFlow st.raw)⟩ : TSyntax `tritonStmt)
        let finite ← FiniteHelperSource.prepare original []
        let statements ← RandomProfile.prepare finite
        let integerPinned := Inference.collectIntPinned statements
        let regions := Inference.collectRegionDTypes statements
        let pointerElements := Inference.collectPtrElems regions statements
        let fp : FPSourceContext := { regions := regions, legacyPointerCasts := Bool.false }
        let (_, code, _, _) ← expandStmts [] [] integerPinned regions pointerElements fp [] [] statements
        let (inputs,outputs) := Metadata.blockRegions original
        let ins := inputs.toArray
        let outs := outputs.toArray
        let result ← `(ComputeKernel.mk [$ins,*] [$outs,*] [$code,*])
        normalizePrecisionScopes none result.raw
      let expression ← elabTerm expanded none
      synthesizeSyntheticMVarsNoPostponing
      instantiateMVars expression

end VeriTile.Bench.Repairs.LlamaFFScalarCast
