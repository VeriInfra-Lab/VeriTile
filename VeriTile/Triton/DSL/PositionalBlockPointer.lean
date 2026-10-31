import VeriTile.Triton.DSL.SourceDType
namespace VeriTile.Triton.DSL.PositionalBlockPointer
open Lean
syntax "tl.make_block_ptr(" tritonExpr ", " "(" tritonExpr,*,? ")" ", " "(" tritonExpr,*,? ")" ", " "("
  tritonExpr,*,? ")" ", " "(" tritonExpr,*,? ")" ", " "(" num,*,? ")" ")" : tritonExpr

partial def normalize (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM normalize)
  match raw with
  | `(tritonExpr| tl.make_block_ptr($base:tritonExpr, ($parent:tritonExpr,*),
      ($strides:tritonExpr,*), ($offsets:tritonExpr,*), ($block:tritonExpr,*), ($order:num,*))) =>
    let baseKw := mkIdent `base
    let shapeKw := mkIdent `shape
    let strideKw := mkIdent `strides
    let offsetKw := mkIdent `offsets
    let blockKw := mkIdent `block_shape
    let orderKw := mkIdent `order
    return (← `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base:tritonExpr,
      $shapeKw:ident=($parent,*), $strideKw:ident=($strides,*), $offsetKw:ident=($offsets,*),
      $blockKw:ident=($block,*), $orderKw:ident=($order,*)))).raw
  | _ => return raw
end VeriTile.Triton.DSL.PositionalBlockPointer
