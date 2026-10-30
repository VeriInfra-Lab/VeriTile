import VeriTile.Triton
open Lean Elab Term
open VeriTile.Triton VeriTile.Triton.DSL
namespace VeriTile.Bench.Repairs.RbeTritonTransform
private partial def nonnegativeWidenings (stx : Syntax) : MacroM Syntax := do
  if stx.getKind == ``tritonMethodCast then
    let args := stx.getArgs
    if args.size == 5 || args.size == 6 then
      let dtype : TSyntax `tritonDType := ⟨args[args.size-2]!⟩
      match dtype with
      | `(tritonDType| tl.int64) =>
        let value : TSyntax `tritonExpr := ⟨args[0]!⟩
        let admitted ← match value with
          | `(tritonExpr| $(($_:term : Nat))) => pure Bool.true
          | `(tritonExpr| tl.program_id(axis=$_:tritonExpr)) => pure Bool.true
          | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr)) => pure Bool.true
          | _ => pure Bool.false
        unless admitted do Macro.throwErrorAt stx "unsupported nonnegative widening operand"
        return ← nonnegativeWidenings value.raw
      | _ => pure ()
  match stx with
  | .node info kind args => return .node info kind (← args.mapM nonnegativeWidenings)
  | _ => return stx

/-- The source's nonnegative size/stride/PID/offset values use the core Nat
index carrier. Keep every int64 cast in the readable quotation; the separate
native-index theorem justifies width erasure. Signed token values remain Int.
The core store constructor uses the value dtype rather than the pointer dtype;
the adapter also reifies exactly Triton's implicit FP32 pointer-element store
conversion. These annotations change no source arithmetic or evaluator. -/
syntax "rbe_triton" "{" tritonStmt* "}" : term
elab_rules : term
  | `(rbe_triton { $stmts:tritonStmt* }) => do
    let expanded ← liftMacroM do
      let statements ← stmts.toList.mapM fun (statement : TSyntax `tritonStmt) => do
        let statement : TSyntax `tritonStmt := ⟨← nonnegativeWidenings statement.raw⟩
        match statement with
        | `(tritonStmt| tl.store($pointer:tritonExpr, $value:tritonExpr, $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
            `(tritonStmt| tl.store($pointer:tritonExpr, ($value:tritonExpr).to(DTYPE), $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
        | _ => pure statement
      let integerPinned := ["start_token_position","start_block","tks"]
      let regions := Inference.collectRegionDTypes statements
      let pointers := Inference.collectPtrElems regions statements
      let fp : FPSourceContext := { regions := regions, legacyPointerCasts := Bool.false }
      let pinned := ["M","K","stride_x_batch","stride_x_m","stride_x_n",
        "stride_out_batch","stride_out_m","stride_out_n","pid_batch","pid","pid_m","pid_n",
        "offs_m","offs_n","freqs"]
      let (_, code, _, _) ← expandStmts [] pinned integerPinned regions pointers fp [] [] statements
      let (inputs,outputs) := Metadata.blockRegions statements
      let ins := inputs.toArray
      let outs := outputs.toArray
      let result ← `(ComputeKernel.mk [$ins,*] [$outs,*] [$code,*])
      normalizePrecisionScopes none result.raw
    let expression ← elabTerm expanded none
    synthesizeSyntheticMVarsNoPostponing
    instantiateMVars expression
end VeriTile.Bench.Repairs.RbeTritonTransform
