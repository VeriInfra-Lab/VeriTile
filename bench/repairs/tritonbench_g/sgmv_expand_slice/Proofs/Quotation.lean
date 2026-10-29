import VeriTile.Triton

open Lean Elab Term
open VeriTile.Triton VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.SgmvExpandSlice

/-! Local adapter for an upstream inference ambiguity, not a new evaluator.
The source explicitly declares all six region types. Every integer value has an explicit source cast or typed integer expression, so
no untyped Nat load/address inference is necessary. We disable only the Nat
*inference hints* for this fully typed source, and use
unchanged upstream statement expansion, FP provenance and precision scoping.
The terminal store is distributed into both paths of the immediately preceding
ADD_INPUTS conditional before dtype inference. This is continuation distribution:
each path executes the same statements and one store, while the false path uses
the dtype of tiled_c before the skipped add. The upstream one-arm conditional
otherwise exports the then-arm dtype even when the source flag is false.
The shared store constructor uses the value's dtype and does not insert Triton's
implicit pointer-element conversion. We therefore reify that existing conversion
on precisely the paths where the computed value dtype differs from the declared
output element dtype. Same-type stores add no identity cast. No source arithmetic,
address, executed store or numerical conversion is added or removed.
The source generator wraps pointer RHSs to select ordinary expression expansion,
not the upstream self-add shorthand. No shared DSL source/object is modified. -/
syntax "sgmv_triton" "{" tritonStmt* "}" : term

private def distributeTerminalStore (statements : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  let regions := Inference.collectRegionDTypes statements
  let some weight := Inference.lookupRegionDType regions "lora_ptr"
    | Macro.throwError "sgmv_triton requires the declared weight element dtype"
  let some output := Inference.lookupRegionDType regions "out_ptr"
    | Macro.throwError "sgmv_triton requires the declared output element dtype"
  unless weight == .fp16 || weight == .bf16 do
    Macro.throwError "sgmv_triton supports only source-required half/bfloat weights"
  let outputSyntax ← match output with
    | .fp16 => `(tritonDType| tl.float16)
    | .bf16 => `(tritonDType| tl.bfloat16)
    | .fp32 => `(tritonDType| tl.float32)
    | _ => Macro.throwError "sgmv_triton requires half/bfloat/single output storage"
  let addition := if output == .fp32 then DInfo.fp32
    else if output == .fp16 || weight == .fp16 then DInfo.fp16 else DInfo.bf16
  match statements.reverse with
  | terminal :: branch :: beforeRev =>
      match terminal,branch with
      | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)),
        `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
          let converted ← `(tritonStmt| tl.store($p:tritonExpr,
            ($v:tritonExpr).to($outputSyntax:tritonDType), $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
          let trueStore := if addition == output then terminal else converted
          let falseStore := if weight == output then terminal else converted
          let yes := body.push trueStore
          let distributed ← `(tritonStmt| if $condition { $yes* } else { $falseStore })
          return beforeRev.reverse ++ [distributed]
      | _,_ => Macro.throwError "sgmv_triton expects a terminal conditional followed by its masked store"
  | _ => Macro.throwError "sgmv_triton expects a complete source body"

elab_rules : term
  | `(sgmv_triton { $stmts:tritonStmt* }) => do
      let expanded ← liftMacroM do
        let original ← stmts.toList.mapM fun st => do
          return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
        let branchLocal ← distributeTerminalStore original
        let finite ← FiniteHelperSource.prepare branchLocal []
        let statements ← RandomProfile.prepare finite
        let pinned : List String := []
        let integerPinned := Inference.collectIntPinned statements
        let regions := Inference.collectRegionDTypes statements
        let pointerElements := Inference.collectPtrElems regions statements
        let fp : FPSourceContext := { regions := regions, legacyPointerCasts := Bool.false }
        let (_, code, _, _) ← expandStmts [] pinned integerPinned regions pointerElements fp [] [] statements
        let (inputs,outputs) := Metadata.blockRegions original
        let ins := inputs.toArray
        let outs := outputs.toArray
        let result ← `(ComputeKernel.mk [$ins,*] [$outs,*] [$code,*])
        normalizePrecisionScopes none result.raw
      let expression ← elabTerm expanded none
      synthesizeSyntheticMVarsNoPostponing
      instantiateMVars expression

end VeriTile.Bench.Repairs.SgmvExpandSlice
