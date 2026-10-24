import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.WordAddQuotation
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.WordAdd

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/-- A symbolic tile fixture binds the exact readable source call to the core
expression. The escaped Lean identifier `«partial»` has exactly the Python name
`partial`; no argument is renamed or reordered. This is not a kernel theorem. -/
def quotedWordAdd (rows columns : Nat) (left right : Int) : ComputeKernel := int8mod_triton {
  accumulator = tl.full([$(rows), $(columns)], $(left), dtype=tl.int32)
  «partial» = tl.full([$(rows), $(columns)], $(right), dtype=tl.int32)
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;",
    constraints="=r,r,r",
    args=[accumulator, «partial»],
    dtype=tl.int32,
    is_pure=True,
    pack=1,
  )
}

theorem quotedWordAdd_body (rows columns : Nat) (left right : Int) :
    (quotedWordAdd rows columns left right).surfaceBody =
      [.assign .int [rows,columns] "accumulator" (.alg (.full [rows,columns] (.constInt left))),
       .assign .int [rows,columns] "partial" (.alg (.full [rows,columns] (.constInt right))),
       .assign .int [rows,columns] "accumulator"
         (.alg (wordAddExpr (.ref .int [rows,columns] "accumulator")
           (.ref .int [rows,columns] "partial")))] := by rfl

#axiomsClean quotedWordAdd_body
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
