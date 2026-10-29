import VeriTile.Triton.DSL.Expansion.Common

/-! Source-preserving parsing of the exact pure single-result add.u32 ABI.
The mathematical instruction is addition modulo 2^32 followed by signed32
readback. Native operand-width/range refinement belongs to the caller; this
frontend neither proves hardware conformance nor accepts arbitrary assembly. -/
namespace VeriTile.Triton.DSL
open Lean

def inlineAddU32Arguments (assembly : TSyntax `str)
    (kwargs : TSyntaxArray `tritonInlineAsmKwarg) :
    MacroM (TSyntax `tritonExpr × TSyntax `tritonExpr) := do
  unless assembly.getString == "add.u32 $0, $1, $2;" do
    Macro.throwError "inline ASM: only the modeled add.u32 instruction is supported"
  let mut seen : List String := []
  let mut operands : Option (TSyntax `tritonExpr × TSyntax `tritonExpr) := none
  for kw in kwargs do
    unless kw.raw.getNumArgs > 0 && kw.raw[0].isIdent do
      Macro.throwError "inline ASM: expected a named literal ABI option"
    let name := kw.raw[0].getId.eraseMacroScopes.toString
    if seen.contains name then Macro.throwError "inline ASM: duplicate ABI option"
    seen := name :: seen
    match name with
    | "constraints" =>
      let `(tritonInlineAsmKwarg| $_:ident = $value:tritonExpr) := kw
        | Macro.throwError "inline ASM: constraints must be the literal =r,r,r"
      let `(tritonExpr| $value:str) := value
        | Macro.throwError "inline ASM: constraints must be the literal =r,r,r"
      unless value.getString == "=r,r,r" do
        Macro.throwError "inline ASM: constraints must be the literal =r,r,r"
    | "args" =>
      let `(tritonInlineAsmKwarg| $_:ident = [$values:tritonExpr,*]) := kw
        | Macro.throwError "inline ASM: args must be two integer register identifiers"
      let values := values.getElems
      unless values.size == 2 do
        Macro.throwError "inline ASM: args must be two integer register identifiers"
      for value in values do
        let `(tritonExpr| $_:ident) := value
          | Macro.throwError "inline ASM: args must be two integer register identifiers"
      operands := some (values[0]!,values[1]!)
    | "dtype" =>
      match kw with
      | `(tritonInlineAsmKwarg| $_:ident = tl.int32) => pure ()
      | _ => Macro.throwError "inline ASM: result dtype must be tl.int32"
    | "is_pure" =>
      let `(tritonInlineAsmKwarg| $_:ident = $value:tritonExpr) := kw
        | Macro.throwError "inline ASM: is_pure must be literal True"
      match value with
      | `(tritonExpr| true) => pure ()
      | `(tritonExpr| $value:ident) =>
        unless value.getId.eraseMacroScopes == `True do
          Macro.throwError "inline ASM: is_pure must be literal True"
      | _ => Macro.throwError "inline ASM: is_pure must be literal True"
    | "pack" =>
      let `(tritonInlineAsmKwarg| $_:ident = $value:tritonExpr) := kw
        | Macro.throwError "inline ASM: pack must be literal 1"
      let `(tritonExpr| $value:num) := value
        | Macro.throwError "inline ASM: pack must be literal 1"
      unless value.getNat == 1 do Macro.throwError "inline ASM: pack must be literal 1"
    | _ => Macro.throwError "inline ASM: unsupported ABI option"
  unless seen.length == 5 do Macro.throwError "inline ASM: complete ABI options are required"
  let some result := operands
    | Macro.throwError "inline ASM: args must be two integer register identifiers"
  return result

end VeriTile.Triton.DSL
