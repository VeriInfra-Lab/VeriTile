import bench.tritonbench_g.int8_matmul_kernel.Proofs.TileConfig

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte DefinedCast)
noncomputable section

/-- The exact frontend storage encoding: FP32 raw bits are values of its
floating carrier, with `.fp32` precision retained by source load/cast syntax. -/
def fp32Cell (word : Word32) : Cell Word32 := .mk .real word
def uint8Cell (byte : Byte) : Cell Word32 := .mk .nat byte.toNat
def int32Cell (word : Word32) : Cell Word32 := .mk .int word.toInt

theorem fp32_read (word : Word32) : (fp32Cell word).read .real = word := by
  simp [fp32Cell]
theorem uint8_read (byte : Byte) : (uint8Cell byte).read .nat = byte.toNat := by
  simp [uint8Cell]
theorem int32_read (word : Word32) : (int32Cell word).read .int = word.toInt := by
  simp [int32Cell]

theorem source_conversion (cast : Word32 → Byte) (rest : Algebra Word32) :
    (backend cast rest).floatToInt (some .fp32) .real 8 = some (convert8 cast) := by
  simp [backend]

theorem source_zero (cast : Word32 → Byte) (rest : Algebra Word32) :
    (backend cast rest).literal (some .fp32) .real 0 = 0 := by simp [backend]

theorem true_conditional_preserved (ins outs : List RegionName) (body : List ComputeStmt) :
    compileAssertionProjection (.mk ins outs (.alg (.ifThen (.constBool Bool.true) []) :: body)) =
      .mk ins outs (.alg (.ifThen (.constBool Bool.true) []) :: body) := by rfl

theorem nonempty_conditional_preserved (ins outs : List RegionName) (stmt : Stmt)
    (inside : List Stmt) (body : List ComputeStmt) :
    compileAssertionProjection
      (.mk ins outs (.alg (.ifThen (.constBool Bool.false) (stmt :: inside)) :: body)) =
      .mk ins outs (.alg (.ifThen (.constBool Bool.false) (stmt :: inside)) :: body) := by rfl

#axiomsClean fp32_read
#axiomsClean uint8_read
#axiomsClean int32_read
#axiomsClean source_conversion
#axiomsClean source_zero
#axiomsClean true_conditional_preserved
#axiomsClean nonempty_conditional_preserved
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
