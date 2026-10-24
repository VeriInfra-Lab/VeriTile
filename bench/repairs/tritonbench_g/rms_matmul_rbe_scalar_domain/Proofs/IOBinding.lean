import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Target
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOByteSourceIteration

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBinding
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
variable (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
variable (epsilon theta : α) (order : List (Nat → Nat))
variable (vals : ArrayValues α (IO.inputs fp8 p)) (s : State α)
variable (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s)
include hin

theorem input_x (phase : Fin 3) (off : Nat) :
    (s.mem (p phase).X off).read .real=IO.xValues fp8 p vals phase off := by
  have h:=hin (IO.xSlot phase) off
  fin_cases phase <;> cases fp8 <;>
    simpa [IO.qkvIO,IO.inputs,IO.channel,IO.xSlot,IO.xValues,mapValue] using h

theorem input_w (phase : Fin 3) (off : Nat) :
    (s.mem (p phase).W off).read (IO.weightDType fp8)=IO.wValues fp8 p vals phase off := by
  have h:=hin (IO.wSlot phase) off
  fin_cases phase <;> cases fp8 <;>
    simpa [IO.qkvIO,IO.inputs,IO.channel,IO.wSlot,IO.wValues,IO.weightDType,mapValue] using h

theorem input_rms (phase : Fin 3) (off : Nat) :
    (s.mem (p phase).RMS off).read (IO.weightDType fp8)=IO.rmsValues fp8 p vals phase off := by
  have h:=hin (IO.rmsSlot phase) off
  fin_cases phase <;> cases fp8 <;>
    simpa [IO.qkvIO,IO.inputs,IO.channel,IO.rmsSlot,IO.rmsValues,IO.weightDType,mapValue] using h

theorem bytes (domain : IO.InputDomain fp8 p vals) (phase : Fin 3) :
    IOBytes.InputBytes fp8 (qkvArgument p phase) s := by
  intro on r member off
  subst fp8
  rcases member with rfl|rfl
  · have hw : (s.mem (p phase).W off).read .int=IO.wValues Bool.true p vals phase off :=
      input_w a native Bool.true p epsilon theta order vals s hin phase off
    rw [show (qkvArgument p phase).W=(p phase).W from rfl,hw]
    exact (domain phase off).1
  · have hr : (s.mem (p phase).RMS off).read .int=IO.rmsValues Bool.true p vals phase off :=
      input_rms a native Bool.true p epsilon theta order vals s hin phase off
    rw [show (qkvArgument p phase).RMS=(p phase).RMS from rfl,hr]
    exact (domain phase off).2

theorem row (phase : Fin 3) (batch row : Nat) :
    rawRow a fp8 (qkvArgument p phase) s batch row=Target.rowInput a fp8 p vals phase batch row := by
  have hx:=input_x a native fp8 p epsilon theta order vals s hin phase
  have hw:=input_w a native fp8 p epsilon theta order vals s hin phase
  have hr:=input_rms a native fp8 p epsilon theta order vals s hin phase
  cases fp8 <;> simp [IO.weightDType] at hw hr
  all_goals simp only [rawRow,Target.rowInput,qkvArgument,Bool.false_eq_true,if_false,if_true,hx,hw,hr]

theorem target (e : IO.Event order) (pid : s.pids=order[e.launch]) :
    calleeTarget a native (selectedRotaryPlan e.phase) fp8 (qkvArgument p e.phase)
      s epsilon theta e.lane=Target.raw a native fp8 p epsilon theta vals e := by
  have hr:=row a native fp8 p epsilon theta order vals s hin e.phase
  cases native <;> simp only [calleeTarget,Bool.false_eq_true,if_false,if_true,
    rawTile,rawValue,nativeRawTile,nativeRawValue,programRow,programCol,pid,hr,
    Target.raw,Target.row,Target.col] <;> rfl

#axiomsClean bytes
#axiomsClean row
#axiomsClean target
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBinding
