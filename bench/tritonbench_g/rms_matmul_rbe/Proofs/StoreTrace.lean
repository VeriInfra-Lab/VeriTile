import bench.tritonbench_g.rms_matmul_rbe.Proofs.Caller
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVReadback

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def row (i : Contract.Inputs) (id : Program) (r : Fin 16) : Nat :=
  id.block/RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)*16+r.val
def col (i : Contract.Inputs) (id : Program) (n : Fin 64) : Nat :=
  id.block%RmsMatmulRbe.tilesN (RmsMatmulRbe.Checked.shape i.original)*64+n.val
def logicalOffset (i : Contract.Inputs) (id : Program) (lane : TileIndex [16,64]) : Nat :=
  let s := RmsMatmulRbe.Checked.shape i.original
  id.batch*(s.M*s.N)+row i id lane.1*s.N+col i id lane.2.1
def activeLane (i : Contract.Inputs) (id : Program) (lane : TileIndex [16,64]) : Prop :=
  row i id lane.1<(RmsMatmulRbe.Checked.shape i.original).M ∧
    col i id lane.2.1<(RmsMatmulRbe.Checked.shape i.original).N
instance (i : Contract.Inputs) (id : Program) (lane : TileIndex [16,64]) :
    Decidable (activeLane i id lane) := inferInstanceAs (Decidable (_ ∧ _))

def lanes (i : Contract.Inputs) (id : Program) :=
  (TileShape.allIndices [16,64]).filter (fun lane => decide (activeLane i id lane))

def outputAddress (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (id : Program) (phase : Fin 3) (lane : TileIndex [16,64]) : Nat :=
  address (RmsMatmulRbe.Checked.output i.original q phase) (logicalOffset i id lane)

def castOutput (a : Arithmetic α) (fp8 : Bool) (value : α) : α :=
  if fp8 then a.cast .real .f8e5 value else a.cast .real .fp16 value

def readOutputCell {α : Type} [Inhabited α] (fp8 : Bool) (cell : Cell α) : α :=
  if fp8 then cell.read .f8e5 else cell.read .fp16

theorem readOutputCell_cast {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool) (value : α) :
    readOutputCell fp8 (outputCell a fp8 value)=castOutput a fp8 value := by
  cases fp8 <;> simp [readOutputCell,outputCell,castOutput]

/-- A local conversion-range law, allowing an exact physical narrow readback
only for the values produced by that one scalar output conversion. -/
def OutputStorable (access : Access α Store Byte) (a : Arithmetic α) : Prop :=
  ∀ fp8 value,access.storable fp8 (castOutput a fp8 value)

theorem outputRegions_injective : Function.Injective outputRegion := by
  intro x y h
  fin_cases x <;> fin_cases y <;> simp_all [outputRegion]

theorem caller_outputs (i : Contract.Inputs) :
    Function.Injective (fun phase => (arguments i phase).OUT) := outputRegions_injective

theorem caller_offset {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program)
    (phase : Fin 3) (lane : TileIndex [16,64]) :
    outputOffset (qkvArgument (arguments i) phase) ((state access zero i q s id).pids 0)
      (programRow (qkvArgument (arguments i) phase) (state access zero i q s id))
      (programCol (qkvArgument (arguments i) phase) (state access zero i q s id)) lane=
      logicalOffset i id lane := by
  simp [outputOffset,qkvArgument,arguments,state,Program.pids,programRow,programCol,columns,
    logicalOffset,row,col,RmsMatmulRbe.tilesN]

theorem caller_active {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (s : Store) (id : Program)
    (phase : Fin 3) (lane : TileIndex [16,64]) :
    outputActive (qkvArgument (arguments i) phase)
      (programRow (qkvArgument (arguments i) phase) (state access zero i q s id))
      (programCol (qkvArgument (arguments i) phase) (state access zero i q s id)) lane ↔
      activeLane i id lane := by
  simp [outputActive,qkvArgument,arguments,state,Program.pids,programRow,programCol,columns,
    activeLane,row,col,RmsMatmulRbe.tilesN]

def storedEvents {α : Type} [Inhabited α] (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (id : Program) (final : State α) : List (Write α) :=
  (List.finRange 3).flatMap (fun phase => (lanes i id).map (fun lane =>
    ⟨qkvOutputFP8 (useFP8 i) phase,outputAddress i q id phase lane,
      readOutputCell (qkvOutputFP8 (useFP8 i) phase)
        (final.mem (outputRegion phase) (logicalOffset i id lane))⟩))

def valueEvents {α : Type} [Inhabited α] (access : Access α Store Byte) (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (initial : Store) (id : Program) (epsilon theta : HostScalar) :
    List (Write α) :=
  (List.finRange 3).flatMap (fun phase => (lanes i id).map (fun lane =>
    ⟨qkvOutputFP8 (useFP8 i) phase,outputAddress i q id phase lane,
      castOutput a (qkvOutputFP8 (useFP8 i) phase)
        (calleeTarget a native (selectedRotaryPlan phase) (useFP8 i) (qkvArgument (arguments i) phase)
          (state access a.zero i q initial id) (materialize.apply epsilon) (materialize.apply theta) lane)⟩))

def programTrace {α : Type} [Inhabited α] (access : Access α Store Byte) (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (initial : Store) (id : Program) (epsilon theta : HostScalar) :
    Option (List (Write α)) :=
  (FP.Structural.run a.model (qkvCode native (useFP8 i) (arguments i))
    (bindScalars (materializedScalars materialize epsilon theta) (state access a.zero i q initial id))).map
      (storedEvents i q id)

def program {α : Type} [Inhabited α] (access : Access α Store Byte) (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (initial : Store) (id : Program) (epsilon theta : HostScalar) :
    Option Store :=
  (programTrace access a materialize native i q initial id epsilon theta).map
    (fun events => PhysicalKernel.run access events initial)

theorem source_store_materialization {α : Type} [Inhabited α]
    (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (id : Program) (epsilon theta : HostScalar) (final : State α)
    (stored : final.mem=(qkvWrite a native (useFP8 i) (arguments i)
      (bindScalars (materializedScalars materialize epsilon theta)
        (state access a.zero i q initial id))
      (materialize.apply epsilon) (materialize.apply theta)).mem) :
    storedEvents i q id final=valueEvents access a materialize native i q initial id epsilon theta := by
  apply List.flatMap_congr
  intro phase _
  apply List.map_congr_left
  intro lane member
  have live : activeLane i id lane := of_decide_eq_true (List.mem_filter.mp member).2
  have active := (caller_active access a.zero i q initial id phase lane).mpr live
  have cell := qkv_readback a native (useFP8 i) (arguments i)
    (bindScalars (materializedScalars materialize epsilon theta) (state access a.zero i q initial id))
    (materialize.apply epsilon) (materialize.apply theta) (caller_outputs i)
    (fun _ => ⟨rfl,rfl⟩) phase lane active
  change (qkvWrite a native (useFP8 i) (arguments i)
    (bindScalars (materializedScalars materialize epsilon theta) (state access a.zero i q initial id))
    (materialize.apply epsilon) (materialize.apply theta)).mem (outputRegion phase)
    (outputOffset (qkvArgument (arguments i) phase) ((state access a.zero i q initial id).pids 0)
      (programRow (qkvArgument (arguments i) phase) (state access a.zero i q initial id))
      (programCol (qkvArgument (arguments i) phase) (state access a.zero i q initial id)) lane)=_ at cell
  rw [caller_offset] at cell
  simp only [stored,cell,readOutputCell_cast]
  rfl

/-- Successful complete quoted/native Q→K→V execution produces exactly the
independent events, with caller-derived geometry and shared scalar objects. -/
theorem programTrace_complete {α : Type} [Inhabited α]
    (access : Access α Store Byte) (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (q : RmsMatmulRbe.Contract.Tensor)
    (initial : Store) (id : Program) (epsilon theta : HostScalar) :
    programTrace access a materialize native i q initial id epsilon theta=
      some (valueEvents access a materialize native i q initial id epsilon theta) := by
  let start := bindScalars (materializedScalars materialize epsilon theta)
    (state access a.zero i q initial id)
  have scalars := scalar_registers materialize epsilon theta (state access a.zero i q initial id)
  obtain ⟨final,success,stored,_⟩ := qkv_complete a z native (useFP8 i) (arguments i) start
    (materialize.apply epsilon) (materialize.apply theta) scalars.1 scalars.2
    (state_int8 access a.zero i q initial id) (arguments_even i checked) (arguments_apart i)
  dsimp only [start] at success
  simp only [programTrace,success,Option.map_some]
  exact congrArg some (source_store_materialization access a materialize native i q initial id epsilon theta final stored)

theorem program_complete {α : Type} [Inhabited α]
    (access : Access α Store Byte) (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (q : RmsMatmulRbe.Contract.Tensor)
    (initial : Store) (id : Program) (epsilon theta : HostScalar) :
    program access a materialize native i q initial id epsilon theta=
      some (PhysicalKernel.run access
        (valueEvents access a materialize native i q initial id epsilon theta) initial) := by
  rw [program,programTrace_complete access a z materialize native i checked]
  rfl

#axiomsClean readOutputCell_cast
#axiomsClean caller_outputs
#axiomsClean caller_offset
#axiomsClean caller_active
#axiomsClean source_store_materialization
#axiomsClean programTrace_complete
#axiomsClean program_complete
#specNonCircular valueEvents avoiding [programTrace,storedEvents,execution,qkvCode]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
