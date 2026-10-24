import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.IO
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicRealIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.LogicalIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original domain and judgment, including typed cells, are retained. -/
theorem logical_iff (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    (io d8 fp8 B p ⊨{domain fp8 B p} fun values batch row col =>
      KernelMath.output p.K p.headDim p.start p.rotary
        (rowInput d8 fp8 B p values batch.val row.val) row.val col.val) ↔
    (PublicIO.io (KernelReal.operations d8) fp8 B p).ImplementsOn (PublicIO.domain fp8 B p)
      (PublicIO.mathematical d8 fp8 B p) := by
  unfold LogicalProgramIO.ImplementsOn
  have domain_eq : (fun raw => domain fp8 B p ((io d8 fp8 B p).inputs raw)) =
      PublicIO.domain fp8 B p := by
    funext raw
    change PublicIO.domain fp8 B p ((inputs fp8 B p).symm ((inputs fp8 B p) raw)) = _
    rw [(inputs fp8 B p).symm_apply_apply]
  rw [domain_eq]
  have target_eq : (io d8 fp8 B p).lift (fun values batch row col =>
      KernelMath.output p.K p.headDim p.start p.rotary
        (rowInput d8 fp8 B p values batch.val row.val) row.val col.val) =
      PublicIO.mathematical d8 fp8 B p := by
    funext raw channel
    fin_cases channel
    funext j
    cases fp8
    · change KernelMath.output p.K p.headDim p.start p.rotary
        (KernelReal.logical d8 Bool.false
          (PublicIO.logical Bool.false B p
            ((inputs Bool.false B p).symm ((inputs Bool.false B p) raw))) j.1.val j.2.1.val)
        j.2.1.val j.2.2.1.val =
        KernelMath.output p.K p.headDim p.start p.rotary
          (KernelReal.logical d8 Bool.false (PublicIO.logical Bool.false B p raw) j.1.val j.2.1.val)
          j.2.1.val j.2.2.1.val
      rw [(inputs Bool.false B p).symm_apply_apply]
    · change KernelMath.output p.K p.headDim p.start p.rotary
        (KernelReal.logical d8 Bool.true
          (PublicIO.logical Bool.true B p
            ((inputs Bool.true B p).symm ((inputs Bool.true B p) raw))) j.1.val j.2.1.val)
        j.2.1.val j.2.2.1.val =
        KernelMath.output p.K p.headDim p.start p.rotary
          (KernelReal.logical d8 Bool.true (PublicIO.logical Bool.true B p raw) j.1.val j.2.1.val)
          j.2.1.val j.2.2.1.val
      rw [(inputs Bool.true B p).symm_apply_apply]

  rw [target_eq]
  rfl

end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.LogicalIO
