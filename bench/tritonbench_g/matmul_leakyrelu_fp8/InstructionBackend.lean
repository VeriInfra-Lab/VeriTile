import bench.tritonbench_g.matmul_leakyrelu_fp8.EventResult
import bench.tritonbench_g.matmul_leakyrelu_fp8.Backend

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8.Checked
noncomputable section

/-- Selected BK64 profile expands structurally to eight ordered fixed K8
instruction lanes, with the supplied accumulator. Unused batch/K shapes have
no new numerical primitive. In particular there is no arbitrary-K oracle. -/
def selectedDot (ops : Operations α) : DotAcc α where
  apply := by
    intro precision left right out options batch m k n a b acc i
    cases batch with
    | nil =>
      exact if hk : k = 64 then
        block ops (fun lane => a (i.1,Fin.cast hk.symm lane,PUnit.unit))
          (fun lane => b (Fin.cast hk.symm lane,i.2.1,PUnit.unit)) (acc i)
        else acc i
    | cons head tail => exact acc i
  le := fun _ _ _ x => ops.nonnegative x

theorem selected_dot (ops : Operations α) (a : TileIndex [128,64] → α)
    (b : TileIndex [64,256] → α) (acc : TileIndex [128,256] → α)
    (i : TileIndex [128,256]) :
    (selectedDot ops).apply (some .fp32) .real .real .real {} (batch := []) a b acc i =
      block ops (fun lane => a (i.1,lane,PUnit.unit))
        (fun lane => b (lane,i.2.1,PUnit.unit)) (acc i) := by
  simp [selectedDot]

/-- Only scalar source operations are left as local model boundaries. -/
structure ScalarLaws (M : Algebra α) (ops : Operations α) : Prop where
  initialZero : M.literal (some .fp32) .real 0 = ops.zero
  /-- The fp32 precision channel loads the existing 32-bit payload unchanged. -/
  loadValue : ∀ x, M.fp32Load x = x
  loadZero : M.fp32Load (M.literal (some .fp32) .real 0.0) = ops.zero
  slope : ∀ x, M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 0.01) x = ops.slopeMultiply x
  halfCast : ∀ x, M.cast none .real .fp16 x = ops.castHalf x

#specNonCircular ScalarLaws avoiding [eventCode,eventRun,logicalReference]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
