import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPrelude
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod

def maxLoop (p : Parameters) : ComputeStmt :=
  ((code p)[13]?).getD (.effectMarker "unreachable")
def quantizeLoop (p : Parameters) : ComputeStmt :=
  ((code p)[16]?).getD (.effectMarker "unreachable")
def limit (p : Parameters) : Op .int [] :=
  match maxLoop p with
  | .forRangeInt _ _ stop _ _ => stop
  | _ => .constInt 0

def scaleAndReset (p : Parameters) : List ComputeStmt := ((code p).drop 14).take 2
def epilogue (p : Parameters) : List ComputeStmt := (code p).drop 17

theorem maxLoop_shape (p : Parameters) : maxLoop p =
    .forRangeInt "k" (.castNatToInt (.constNat 0)) (limit p) (.constNat 1) (maxBody p) := rfl

theorem quantizeLoop_shape (p : Parameters) : quantizeLoop p =
    .forRangeInt "k" (.castNatToInt (.constNat 0)) (limit p) (.constNat 1) (quantizeBody p) := rfl

theorem code_decomposition (p : Parameters) : code p =
    prelude p ++ [maxLoop p] ++ scaleAndReset p ++ [quantizeLoop p] ++ epilogue p := rfl

theorem limit_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    evalOp M none (limit p) s=some (fun _=>(tripCount p : Int)) := by
  simp only [limit,maxLoop,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  simp [evalOp_unfold,FP.Structural.bop,numeric,NumericDType.sub,Option.bind,
    ComputeDType.eraseDType,tripCount,integralFloorDiv,integralMod]
  funext i
  simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,cdiv_signed_int,
    ← Int.natCast_ediv]

theorem signedLoop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (body : List ComputeStmt) (s : State α) :
    run M [.forRangeInt "k" (.castNatToInt (.constNat 0)) (limit p) (.constNat 1) body] s =
      rangeInt M "k" 0 (tripCount p : Int) 1 body s := by
  rw [run,step]
  have zero : evalOp M none (Op.castNatToInt (Op.constNat 0)) s=some (fun _=>(0 : Int)) := rfl
  have one : evalOp M none (Op.constNat 1) s=some (fun _=>(1 : Nat)) := rfl
  rw [zero,limit_eval,one]
  have nil_run : run M []=some := by funext t;rw [run]
  rw [nil_run]
  change (rangeInt M "k" 0 (tripCount p : Int) 1 body s).bind (fun t=>some t)=_
  cases rangeInt M "k" 0 (tripCount p : Int) 1 body s <;> rfl

theorem maxLoop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M [maxLoop p] s=rangeInt M "k" 0 (tripCount p : Int) 1 (maxBody p) s := by
  rw [maxLoop_shape];exact signedLoop_run M p (maxBody p) s

theorem quantizeLoop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M [quantizeLoop p] s=rangeInt M "k" 0 (tripCount p : Int) 1 (quantizeBody p) s := by
  rw [quantizeLoop_shape];exact signedLoop_run M p (quantizeBody p) s

#axiomsClean code_decomposition
#axiomsClean limit_eval
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
