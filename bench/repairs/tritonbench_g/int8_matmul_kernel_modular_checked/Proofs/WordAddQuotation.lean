import VeriTile.Triton

open Lean Elab Term
open VeriTile.Triton VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked

/-! Task-local syntax for the one exact source inline-assembly call. No shared
parser, core expression, arithmetic evaluator, or compiler rule is changed.
The source spelling survives in the quotation. Its fixed instruction ABI is
checked before expansion to existing explicit unsigned32/signed32 casts.
The separate word-add theorem proves that expression's lane semantics. -/
syntax (name := wordAddAssembly) "tl.inline_asm_elementwise(" str ","
  ident "=" str "," ident "=" "[" tritonExpr "," tritonExpr "]" ","
  ident "=" tritonDType "," ident "=" ident "," ident "=" num "," ")" : tritonExpr

private partial def expandWordAdd (stx : Syntax) : MacroM (Syntax × Nat) := do
  if stx.getKind == ``wordAddAssembly then
    let expression : TSyntax `tritonExpr := ⟨stx⟩
    match expression with
    | `(tritonExpr| tl.inline_asm_elementwise($instruction:str,
        $constraintsKey:ident=$constraints:str, $argsKey:ident=[$lhs:tritonExpr, $rhs:tritonExpr],
        $dtypeKey:ident=$dtype:tritonDType, $purityKey:ident=$purity:ident, $packKey:ident=$pack:num,)) =>
      unless constraintsKey.getId == `constraints && argsKey.getId == `args &&
          dtypeKey.getId == `dtype && purityKey.getId == `is_pure && packKey.getId == `pack do
        Macro.throwErrorAt expression "word-add ABI requires exactly constraints, args, dtype, is_pure, pack in source order"
      unless instruction.getString == "add.u32 $0, $1, $2;" do
        Macro.throwErrorAt instruction "only the exact add.u32 instruction is supported"
      unless constraints.getString == "=r,r,r" do
        Macro.throwErrorAt constraints "word-add ABI requires one output and two input 32-bit registers"
      unless purity.getId == `True && pack.getNat == 1 do
        Macro.throwErrorAt expression "word-add ABI requires is_pure=True and pack=1"
      match dtype, lhs, rhs with
      | `(tritonDType| tl.int32), `(tritonExpr| accumulator), `(tritonExpr| «partial») =>
        let result ← `(tritonExpr| (($lhs:tritonExpr + $rhs:tritonExpr).to(tl.uint32)).to(tl.int32))
        return (result.raw, 1)
      | _, _, _ =>
        Macro.throwErrorAt expression "word-add ABI requires dtype=tl.int32 and args=[accumulator, partial]"
    | _ => Macro.throwErrorAt expression "unsupported word-add source syntax"
  match stx with
  | .node info kind children =>
    let converted ← children.mapM expandWordAdd
    return (.node info kind (converted.map Prod.fst),
      converted.foldl (fun count item => count + item.2) 0)
  | _ => return (stx, 0)

syntax "int8mod_triton" "{" tritonStmt* "}" : term

elab_rules : term
  | `(int8mod_triton { $statements:tritonStmt* }) => do
    let expanded ← liftMacroM do
      let converted ← statements.mapM fun statement => expandWordAdd statement.raw
      unless converted.foldl (fun count item => count + item.2) 0 == 1 do
        Macro.throwError "int8mod_triton requires exactly one source word-add call"
      let body : Array (TSyntax `tritonStmt) := converted.map fun item => ⟨item.1⟩
      `(triton { $body* })
    let expression ← elabTerm expanded none
    synthesizeSyntheticMVarsNoPostponing
    instantiateMVars expression

end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
