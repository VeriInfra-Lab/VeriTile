import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelGemmSource
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ControlArithmetic
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
open VeriTile.Triton FP.Structural KernelExecution ControlArithmetic
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod

def stop (p : Parameters) : Op .int [] :=
  match iteration p with
  | .forRangeInt _ _ limit _ _ => limit
  | _ => .constInt 0

theorem iteration_shape (p : Parameters) : iteration p =
    .forRangeInt "k" (.castNatToInt (.constNat 0)) (stop p) (.constNat 1) (body p) := rfl

theorem stop_eval {F : Type} [Inhabited F] (M : Algebra F) (p : Parameters) (s : State F) :
    evalOp M none (stop p) s=some (fun _ => (tripCount p : Int)) := by
  simp only [stop, iteration, source, KernelSource.gemmHalfHalfKernel, gemmHalfHalfKernel,
    ComputeKernel.surfaceBody, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  simp [evalOp_unfold, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, numeric, NumericDType.sub,
    Option.bind, ComputeDType.eraseDType, tripCount, integralFloorDiv, integralMod]
  funext i
  simp only [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  simp only [← Int.natCast_mul, cdiv_signed_int, ← Int.natCast_ediv]

theorem iteration_run {I F H : Type} [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans) (p : Parameters) (s : State F) :
    KernelExecution.run M carrier ops plans [iteration p] s =
      KernelExecution.rangeInt M carrier ops plans "k" 0 (tripCount p : Int) 1 (body p) s := by
  rw [KernelExecution.run, iteration_shape, KernelExecution.step]
  have zero : evalOp M none (Op.castNatToInt (Op.constNat 0)) s=some (fun _=>(0 : Int)) := rfl
  have one : evalOp M none (Op.constNat 1) s=some (fun _=>(1 : Nat)) := rfl
  rw [zero, stop_eval, one]
  have nil_run : KernelExecution.run M carrier ops plans [] = some := by
    funext t
    rw [KernelExecution.run]
  rw [nil_run]
  change (KernelExecution.rangeInt M carrier ops plans "k" 0 (tripCount p : Int) 1 (body p) s).bind (fun t => some t) = _
  cases KernelExecution.rangeInt M carrier ops plans "k" 0 (tripCount p : Int) 1 (body p) s <;> rfl

theorem iteration_step {I F H : Type} [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans) (p : Parameters) (s : State F) :
    KernelExecution.step M carrier ops plans (iteration p) s =
      KernelExecution.rangeInt M carrier ops plans "k" 0 (tripCount p : Int) 1 (body p) s := by
  rw [iteration_shape,KernelExecution.step]
  have zero : evalOp M none (Op.castNatToInt (Op.constNat 0)) s=some (fun _=>(0 : Int)) := rfl
  have one : evalOp M none (Op.constNat 1) s=some (fun _=>(1 : Nat)) := rfl
  rw [zero,stop_eval,one]
  rfl

#axiomsClean iteration_run
#axiomsClean iteration_step
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
