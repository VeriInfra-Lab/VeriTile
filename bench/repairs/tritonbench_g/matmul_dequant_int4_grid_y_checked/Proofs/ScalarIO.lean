import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.PublicIO
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.NativeResults

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

theorem difference_bounds {α : Type} {f : Source.Profile} {p : Source.Parameters}
    (xs : Inputs α f p) (x : KernelGrid.Coordinate p) :
    -15≤difference xs x ∧ difference xs x≤15 := by
  have a := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.mask_bound
    (BitVec.ofInt 32 (packed xs Bool.false x >>> (x.1.val%8*4)))
  have b := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.mask_bound
    (BitVec.ofInt 32 (packed xs Bool.true x >>> (x.2.val%8*4)))
  change 0≤Source.nibble (packed xs Bool.false x) x.1.val ∧
    Source.nibble (packed xs Bool.false x) x.1.val≤15 at a
  change 0≤Source.nibble (packed xs Bool.true x) x.2.val ∧
    Source.nibble (packed xs Bool.true x) x.2.val≤15 at b
  unfold difference
  omega

theorem scalar_real (f : Source.Profile) (p : Source.Parameters)
    (xs : Inputs ℝ f p) (x : KernelGrid.Coordinate p) :
    scalarFormula f Spec.real p xs x=(difference xs x : ℝ)*scale f xs x := by
  cases f <;> rfl

theorem scalar_native (f : Source.Profile) (I : Native.Instructions)
    (conversion : f=.half → Native.Conversion I) (p : Source.Parameters)
    (xs : Inputs Native.Word f p) (x : KernelGrid.Coordinate p) :
    scalarFormula f (Native.scalar I) p xs x=
      match f with
      | .half => (I.mul16 (I.convert16 .signed (difference xs x))
          ((scale .half xs x).truncate 16)).zeroExtend 32
      | .single => I.mul32 (I.convert32 .signed (difference xs x)) (scale .single xs x) := by
  cases f with
  | half =>
    have h := conversion rfl _ (difference_bounds xs x).1 (difference_bounds xs x).2
    simp [scalarFormula,Native.scalar,Source.Profile.dtype,h]
  | single => rfl

#axiomsClean scalar_native
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO
