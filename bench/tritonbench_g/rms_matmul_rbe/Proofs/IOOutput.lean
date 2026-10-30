import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOGrid
import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOCoverage

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOOutput
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def expected (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) (vals : ArrayValues α (IO.inputs fp8 p)) :
    ChannelValues α (IO.outputs fp8 p order) :=
  Target.result fp8 p order fun e=>a.cast .real (if qkvOutputFP8 fp8 e.phase then .f8e5 else .fp16)
    (Target.raw a native fp8 p epsilon theta vals e)

theorem expected_cell (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) (vals : ArrayValues α (IO.inputs fp8 p))
    (phase : Fin 3) (j : TileIndex (IO.outputs fp8 p order)[phase].shape) :
    Cell.mk (IO.outputs fp8 p order)[phase].dtype (expected a native fp8 p epsilon theta order vals phase j)=
      IOEvents.value a native fp8 p epsilon theta vals
        ⟨(IO.outputIndex fp8 p order phase j).1,phase,(IO.outputIndex fp8 p order phase j).2⟩ := by
  fin_cases phase <;> cases fp8 <;> rfl

theorem output (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) (vals : ArrayValues α (IO.inputs fp8 p))
    (initial final : State α)
    (memory : final.mem=((IO.events order).foldl (IOEvents.put a native fp8 p epsilon theta vals) initial).mem) :
    (IO.qkvIO a native fp8 p epsilon theta order).OutputBinding vals
      (expected a native fp8 p epsilon theta order vals) final := by
  intro phase j winner
  change Fin 3 at phase
  let lane:=IO.outputIndex fp8 p order phase j
  let e : IO.Event order:=⟨lane.1,phase,lane.2⟩
  have hw : IO.lastAt p order (IO.eventRegion p e) (IO.eventOffset p e)=some e := winner
  have read : final.mem (IO.eventRegion p e) (IO.eventOffset p e)=IOEvents.value a native fp8 p epsilon theta vals e := by
    rw [memory,IOEvents.scatter_last,hw]
    rfl
  have cell:=expected_cell a native fp8 p epsilon theta order vals phase j
  change mapCell id (final.mem (IO.outputs fp8 p order)[phase].buf (IO.eventOffset p e))=_
  rw [IO.output_channel]
  change mapCell id (final.mem (IO.eventRegion p e) (IO.eventOffset p e))=_
  rw [read]
  have idValue : mapCell id (IOEvents.value a native fp8 p epsilon theta vals e)=
      IOEvents.value a native fp8 p epsilon theta vals e := by
    unfold IOEvents.value outputCell
    split <;> rfl
  exact idValue.trans cell.symm

theorem frame (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) (vals : ArrayValues α (IO.inputs fp8 p))
    (initial final : State α)
    (memory : final.mem=((IO.events order).foldl (IOEvents.put a native fp8 p epsilon theta vals) initial).mem) :
    (IO.qkvIO a native fp8 p epsilon theta order).Frame vals initial final := by
  intro r off outside
  rw [memory,IOEvents.scatter_last,IOCoverage.outside_none a native fp8 p epsilon theta order vals r off outside]
  rfl

theorem correctness (a : Arithmetic α) (zero : ZeroLaws a) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
    (vals : ArrayValues α (IO.inputs fp8 p)) (domain : IO.InputDomain fp8 p vals)
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (initial : State α) (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals initial) :
    ∃ final,runSchedule ((IO.qkvIO a native fp8 p epsilon theta order).schedule vals) initial=some final ∧
      (IO.qkvIO a native fp8 p epsilon theta order).OutputBinding vals
        (expected a native fp8 p epsilon theta order vals) final ∧
      (IO.qkvIO a native fp8 p epsilon theta order).Frame vals initial final := by
  obtain ⟨final,ran,memory,_⟩:=IOGrid.complete a zero native fp8 p epsilon theta order vals domain even apart initial hin
  exact ⟨final,ran,output a native fp8 p epsilon theta order vals initial final memory,
    frame a native fp8 p epsilon theta order vals initial final memory⟩

theorem expected_real (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat)) (vals : ArrayValues ℝ (IO.inputs fp8 p)) :
    expected mathArithmetic Bool.false fp8 p epsilon.value theta.value order vals=
      Target.mathematical fp8 p epsilon theta order vals := by
  unfold expected Target.mathematical
  apply congrArg (Target.result fp8 p order)
  funext e
  change referenceValue realOps _ _ _ _ _ _ _=realTarget _ _ _ _ _
  exact reference_real _ _ _ _ _ _ _

theorem expected_native (instructions : Native.Primitives) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : Native.Word) (order : List (Nat → Nat))
    (vals : ArrayValues Native.Word (IO.inputs fp8 p)) :
    expected (Native.arithmetic instructions) Bool.true fp8 p epsilon theta order vals=
      Target.native instructions fp8 p epsilon theta order vals := rfl

#axiomsClean correctness
#axiomsClean expected_real
#axiomsClean expected_native
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOOutput
