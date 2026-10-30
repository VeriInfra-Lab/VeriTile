import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelExecution
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IntegerDotPlanBridge
import VeriTile.Triton.Float.IntegerDotIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.SharedInteger
open VeriTile.Triton FP.Structural
open KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000
namespace P
export VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote.SharedInteger
  (plan arithmetic plan_eval)
end P

def representation (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) :
    Int8DequantMatmul.ScalarCast.Quote.IntegerModel I F H where
  ops:=ops
  encode:=carrier.encode
  decode:=carrier.decode
  roundtrip:=carrier.roundtrip
  zero:=zero
  plans:=plans

def context (plans : Plans) : IntegerDot.Context where
  rows:=0
  cols:=0
  rowRegister:="pid_m"
  columnRegister:="pid_n"
  iterationRegister:="k"
  splitAxis:=1
  plans:=fun k r c split iteration=>P.plan (plans k r c split iteration)
  mode:=.accumulatorNat

def arithmetic (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) : IntegerDot.Arithmetic :=
  P.arithmetic (representation carrier ops zero plans)

theorem accumulate_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    {m k n : Nat} (left : Op .int [m,k]) (right : Op .int [k,n]) (old : Op .int [m,n])
    (s : State F) :
    IntegerDot.accumulateNat M (arithmetic carrier ops zero plans) (context plans) left right old s=
      accumulate M carrier ops plans left right old s := by
  simp only [IntegerDot.accumulateNat,accumulate,context,arithmetic,P.plan_eval,representation]

theorem accumulatorStep_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (statement : ComputeStmt) (s : State F) :
    IntegerDot.accumulatorStep M (arithmetic carrier ops zero plans) (context plans) statement s=
      localStep M carrier ops plans statement s := by
  have same : @IntegerDot.accumulateNat F _ M (arithmetic carrier ops zero plans) (context plans)=
      @KernelExecution.accumulate I F H _ M carrier ops plans := by
    funext m k n left right old state
    exact accumulate_eq M carrier ops zero plans left right old state
  delta IntegerDot.accumulatorStep KernelExecution.localStep
  rw [same]
  rfl

theorem localStep_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (statement : ComputeStmt) (s : State F) :
    IntegerDot.arithmeticStep M (arithmetic carrier ops zero plans) (context plans) statement s=
      localStep M carrier ops plans statement s := by
  delta IntegerDot.arithmeticStep
  split
  · exact accumulatorStep_eq M carrier ops zero plans _ s
  · rename_i noAdd
    delta KernelExecution.localStep
    split
    · exact (noAdd _ _ _ _ _ _ _ rfl).elim
    · rfl

mutual
theorem step_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (statement : ComputeStmt) (s : State F) :
    IntegerDot.step M (arithmetic carrier ops zero plans) (context plans) statement s=
      step M carrier ops plans statement s := by
  cases statement
  case forRange name start stop stride body=>
    simp only [IntegerDot.step_unfold,KernelExecution.step.eq_def]
    exact range_eq M carrier ops zero plans name start stop stride body s
  case ifThenElse cond yes no=>
    simp only [IntegerDot.step_unfold,KernelExecution.step.eq_def]
    cases h : evalOp M none cond s with
    | none=>rfl
    | some value=>
      change (if value PUnit.unit then _ else _)=_
      split
      · rename_i active
        simpa [active] using run_eq M carrier ops zero plans yes s
      · rename_i inactive
        simpa [inactive] using run_eq M carrier ops zero plans no s
  case ifThen cond body=>
    simp only [IntegerDot.step_unfold,KernelExecution.step.eq_def]
    rfl
  all_goals simp only [IntegerDot.step_unfold,KernelExecution.step.eq_def]
  all_goals exact localStep_eq M carrier ops zero plans _ s
termination_by (sizeOf statement,0)
decreasing_by
  all_goals simp_wf
  all_goals try omega
  all_goals have string_positive : ∀ n : RegName,0<sizeOf n := by intro n;cases n;simp
  all_goals rename_i idx start stop stride body _ _
  all_goals have hsize:=string_positive idx
  all_goals omega

theorem run_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (code : List ComputeStmt) (s : State F) :
    IntegerDot.run M (arithmetic carrier ops zero plans) (context plans) code s=
      run M carrier ops plans code s := by
  cases code with
  | nil=>simp only [IntegerDot.run,KernelExecution.run]
  | cons statement rest=>
    rw [IntegerDot.run,KernelExecution.run,step_eq]
    congr 1
    funext t
    exact run_eq M carrier ops zero plans rest t
termination_by (sizeOf code,0)
decreasing_by all_goals (simp_wf;omega)

theorem range_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (name : RegName) (start stop stride : Nat) (body : List ComputeStmt) (s : State F) :
    IntegerDot.range M (arithmetic carrier ops zero plans) (context plans) name start stop stride body s=
      range M carrier ops plans name start stop stride body s := by
  rw [IntegerDot.range,KernelExecution.range]
  by_cases hz : stride=0
  · simp [hz]
  · have positive : 0<stride:=Nat.pos_of_ne_zero hz
    by_cases active : start<stop
    · simp only [positive,active,and_self,if_true,hz,if_false]
      rw [run_eq]
      congr 1
      funext t
      exact range_eq M carrier ops zero plans name (start+stride) stop stride body t
    · simp [positive,active,hz]
termination_by (sizeOf body+1,stop-start)
decreasing_by all_goals omega
end

theorem exec_eq [Inhabited F] (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (zero : carrier.decode ops.izero=0) (plans : Plans)
    (source : ComputeKernel) (s : State F) :
    IntegerDot.exec M (arithmetic carrier ops zero plans) (context plans) source s=
      exec M carrier ops plans source s := run_eq M carrier ops zero plans source.surfaceBody s

#axiomsClean accumulate_eq
#axiomsClean localStep_eq
#axiomsClean exec_eq
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.SharedInteger
