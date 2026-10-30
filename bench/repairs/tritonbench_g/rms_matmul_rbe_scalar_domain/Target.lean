import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.IO

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Target
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def row (p : Fin 3 → Arguments) (e : IO.Event order) : Nat :=
  order[e.launch] 1/columns (p e.phase)*16+e.lane.1.val
def col (p : Fin 3 → Arguments) (e : IO.Event order) : Nat :=
  order[e.launch] 1%columns (p e.phase)*64+e.lane.2.1.val

/-- Scalar-array target for each write event. Ordered native K8 instructions,
the original reduction tree and selected rotary FMA order remain explicit. -/
def raw {α : Type} (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (vals : ArrayValues α (IO.inputs fp8 p)) (e : IO.Event order) : α :=
  let arg:=qkvArgument p e.phase
  let inputs:=rowInput a fp8 p vals e.phase (order[e.launch] 0) (row p e)
  if native then nativeReferenceValue a (selectedRotaryPlan e.phase) (parameters arg epsilon theta)
    inputs arg.rotary (row p e) (col p e)
  else referenceValue a.toScalar (parameters arg epsilon theta) inputs
    (fun _ _=>a.plan) a.reduction arg.rotary (row p e) (col p e)

def outputValue {α : Type} (fp8 : Bool) (phase : Fin 3) (x : α) : Value α (IO.outputDType fp8 phase) := by
  unfold IO.outputDType
  split <;> exact x

def result {α : Type} (fp8 : Bool) (p : Fin 3 → Arguments) (order : List (Nat → Nat))
    (f : IO.Event order → α) : ChannelValues α (IO.outputs fp8 p order) := by
  intro o j
  change Fin 3 at o
  let lane:=IO.outputIndex fp8 p order o j
  exact cast (congrArg (Value α) (congrArg IOChannel.dtype (IO.output_channel fp8 p order o)).symm)
    (outputValue fp8 o (f ⟨lane.1,o,lane.2⟩))

/-- Independent RMS-normalized matrix product and adjacent-pair rotary for
each final Q/K/V writer. No source recurrence or execution result defines it. -/
def mathematical (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat)) (vals : ArrayValues ℝ (IO.inputs fp8 p)) :
    ChannelValues ℝ (IO.outputs fp8 p order) :=
  result fp8 p order fun e=>
    realTarget (parameters (p e.phase) epsilon.value theta.value)
      (rowInput mathArithmetic fp8 p vals e.phase (order[e.launch] 0) (row p e))
      (decide (e.phase.val<2)) (row p e) (col p e)

def native (instructions : Native.Primitives) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : Native.Word) (order : List (Nat → Nat))
    (vals : ArrayValues Native.Word (IO.inputs fp8 p)) : ChannelValues Native.Word (IO.outputs fp8 p order) :=
  result fp8 p order fun e=>
    (Native.arithmetic instructions).cast .real
      (if qkvOutputFP8 fp8 e.phase then .f8e5 else .fp16)
      (raw (Native.arithmetic instructions) Bool.true fp8 p epsilon theta vals e)

#specNonCircular mathematical avoiding [State,run,RawGrid.execute,rawTile,sourceTileValues,nativeTileValues]
#specNonCircular native avoiding [State,run,RawGrid.execute,rawTile,nativeRawTile,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Target
