import VeriTile.Triton.DSL.CallSignature

/-! Shared surface syntax. Helper-specific modules register their existing
typed elaborators without changing the caller frontend selected by imports. -/
namespace VeriTile.Triton.DSL.SourceProcedures
syntax (name := sourceParams) "with_triton_params " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" (" fp32_constants " "[" ident,* "]")?
  " in " "triton " RawSourceText.body : term
end VeriTile.Triton.DSL.SourceProcedures

namespace VeriTile.Triton.DSL.SourceSyntax
open Lean

syntax (name := typedParams) "with_triton_params " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" &"signature" term:max
  " in " "triton " RawSourceText.body : term
syntax (name := rebindingParams) "with_triton_params " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" &"rebinding_signature" term:max
  " in " "triton " RawSourceText.body : term
syntax (name := importedParams) "with_triton_imports " "[" tritonCallBinding,* "]"
  " in " "with_triton_params " "(" ident,* ")" " constexpr_args " "[" "]"
  " in " "triton " RawSourceText.body : term

def attachBody (base source : Syntax) : Syntax :=
  base.setArgs (base.getArgs.set! (base.getArgs.size - 1) source.getArgs.back!)

end VeriTile.Triton.DSL.SourceSyntax
