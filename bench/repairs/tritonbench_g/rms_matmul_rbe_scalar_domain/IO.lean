import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.RawGrid
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeArithmetic
import VeriTile.Triton.Float.ArrayProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Named source input arrays and exact ordered Q/K/V writes. Input arrays may
alias. Outputs may alias each other, so only the last source writer at a cell
is an observable output lane; its actual FP16/E5 tag is preserved. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def weightDType (fp8 : Bool) : TileDType := if fp8 then .int else .fp16
def outputDType (fp8 : Bool) (phase : Fin 3) : TileDType :=
  if qkvOutputFP8 fp8 phase then .f8e5 else .fp16

def channel (fp8 : Bool) (p : Fin 3 → Arguments) (i : Fin 9) : ArrayIOChannel :=
  if h : i.val<3 then ⟨.real,(p ⟨i.val,h⟩).X⟩
  else if h : i.val<6 then ⟨weightDType fp8,(p ⟨i.val-3,by omega⟩).W⟩
  else ⟨weightDType fp8,(p ⟨i.val-6,by omega⟩).RMS⟩

def inputs (fp8 : Bool) (p : Fin 3 → Arguments) : List ArrayIOChannel := List.ofFn (channel fp8 p)
def xSlot (phase : Fin 3) : Fin 9 := ⟨phase.val,by omega⟩
def wSlot (phase : Fin 3) : Fin 9 := ⟨phase.val+3,by omega⟩
def rmsSlot (phase : Fin 3) : Fin 9 := ⟨phase.val+6,by omega⟩

@[simp] theorem channel_x (fp8 : Bool) (p : Fin 3 → Arguments) (phase : Fin 3) :
    (inputs fp8 p)[xSlot phase]=⟨.real,(p phase).X⟩ := by
  fin_cases phase <;> rfl
@[simp] theorem channel_w (fp8 : Bool) (p : Fin 3 → Arguments) (phase : Fin 3) :
    (inputs fp8 p)[wSlot phase]=⟨weightDType fp8,(p phase).W⟩ := by
  fin_cases phase <;> rfl
@[simp] theorem channel_rms (fp8 : Bool) (p : Fin 3 → Arguments) (phase : Fin 3) :
    (inputs fp8 p)[rmsSlot phase]=⟨weightDType fp8,(p phase).RMS⟩ := by
  fin_cases phase <;> rfl

def xValues {α : Type} (fp8 : Bool) (p : Fin 3 → Arguments)
    (vals : ArrayValues α (inputs fp8 p)) (phase : Fin 3) (off : Nat) : α :=
  cast (congrArg (Value α) (congrArg ArrayIOChannel.dtype (channel_x fp8 p phase)))
    (vals (xSlot phase) off)
def wValues {α : Type} (fp8 : Bool) (p : Fin 3 → Arguments)
    (vals : ArrayValues α (inputs fp8 p)) (phase : Fin 3) (off : Nat) : Value α (weightDType fp8) :=
  cast (congrArg (Value α) (congrArg ArrayIOChannel.dtype (channel_w fp8 p phase)))
    (vals (wSlot phase) off)
def rmsValues {α : Type} (fp8 : Bool) (p : Fin 3 → Arguments)
    (vals : ArrayValues α (inputs fp8 p)) (phase : Fin 3) (off : Nat) : Value α (weightDType fp8) :=
  cast (congrArg (Value α) (congrArg ArrayIOChannel.dtype (channel_rms fp8 p phase)))
    (vals (rmsSlot phase) off)

def xReads (p : Arguments) (order : List (Nat → Nat)) : List Nat :=
  order.flatMap fun pid=>(List.range 16).flatMap fun m=>(List.range p.K).map fun k=>
    pid 0*p.sxb+((pid 1/columns p*16+m)%p.M)*p.sxm+k*p.sxk
def wReads (p : Arguments) (order : List (Nat → Nat)) : List Nat :=
  order.flatMap fun pid=>(List.range 64).flatMap fun n=>(List.range p.K).map fun k=>
    k*p.swk+((pid 1%columns p*64+n)%p.N)*p.swn
def rmsReads (p : Arguments) (order : List (Nat → Nat)) : List Nat :=
  order.flatMap fun _=>(List.range p.K).map fun k=>k*p.srms

structure Event (order : List (Nat → Nat)) where
  launch : Fin order.length
  phase : Fin 3
  lane : TileIndex [16,64]
  deriving DecidableEq

def events (order : List (Nat → Nat)) : List (Event order) :=
  (List.finRange order.length).flatMap fun launch=>(List.finRange 3).flatMap fun phase=>
    (TileShape.allIndices [16,64]).map fun lane=>⟨launch,phase,lane⟩
def eventRegion (p : Fin 3 → Arguments) (e : Event order) : RegionName := (p e.phase).OUT
def eventOffset (p : Fin 3 → Arguments) (e : Event order) : Nat :=
  outputOffset (qkvArgument p e.phase) (order[e.launch] 0)
    (order[e.launch] 1/columns (qkvArgument p e.phase))
    (order[e.launch] 1%columns (qkvArgument p e.phase)) e.lane
def eventActive (p : Fin 3 → Arguments) (e : Event order) : Prop :=
  outputActive (qkvArgument p e.phase)
    (order[e.launch] 1/columns (qkvArgument p e.phase))
    (order[e.launch] 1%columns (qkvArgument p e.phase)) e.lane
instance (p : Fin 3 → Arguments) (e : Event order) : Decidable (eventActive p e) :=
  inferInstanceAs (Decidable (outputActive _ _ _ _))

def lastAt (p : Fin 3 → Arguments) (order : List (Nat → Nat)) (r : RegionName) (off : Nat) :
    Option (Event order) :=
  (events order).reverse.find? (fun e=>decide (eventActive p e ∧ eventRegion p e=r ∧ eventOffset p e=off))
def Winner (p : Fin 3 → Arguments) (e : Event order) : Prop :=
  lastAt p order (eventRegion p e) (eventOffset p e)=some e

abbrev OutputShape (order : List (Nat → Nat)) : TileShape := [order.length,16,64]
def outputs (fp8 : Bool) (p : Fin 3 → Arguments) (order : List (Nat → Nat)) : List IOChannel :=
  List.ofFn (fun phase : Fin 3=>⟨outputDType fp8 phase,OutputShape order,(p phase).OUT⟩)

@[simp] theorem output_channel (fp8 : Bool) (p : Fin 3 → Arguments) (order : List (Nat → Nat))
    (phase : Fin 3) :
    (outputs fp8 p order)[phase]=⟨outputDType fp8 phase,OutputShape order,(p phase).OUT⟩ := by
  fin_cases phase <;> rfl

def outputIndex (fp8 : Bool) (p : Fin 3 → Arguments) (order : List (Nat → Nat))
    (phase : Fin 3) (j : TileIndex (outputs fp8 p order)[phase].shape) : TileIndex (OutputShape order) :=
  cast (congrArg TileIndex (congrArg IOChannel.shape (output_channel fp8 p order phase))) j

def job {α : Type} (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (pid : Nat → Nat) : Invocation α where
  kernel:=.mk ((List.finRange 3).flatMap fun phase=>[(p phase).X,(p phase).W,(p phase).RMS])
    ((List.finRange 3).map fun phase=>(p phase).OUT) (qkvCode native fp8 p)
  backend:=a.model
  arguments:=[⟨"EPS",.real,[],fun _=>epsilon⟩,⟨"THETA",.real,[],fun _=>theta⟩]
  clearRegisters:=Bool.true
  pids:=pid

def qkvIO {α : Type} (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) : ArrayProgramIO α α where
  inputs:=inputs fp8 p
  outputs:=outputs fp8 p order
  scratch:=[]
  buffers:=(List.finRange 3).flatMap fun phase=>[(p phase).X,(p phase).W,(p phase).RMS,(p phase).OUT]
  reads:=by
    intro i _
    change Fin 9 at i
    exact if h : i.val<3 then xReads (p ⟨i.val,h⟩) order
      else if h : i.val<6 then wReads (p ⟨i.val-3,by omega⟩) order
      else rmsReads (p ⟨i.val-6,by omega⟩) order
  write:=by
    intro o _ j
    change Fin 3 at o
    let lane:=outputIndex fp8 p order o j
    exact eventOffset p ⟨lane.1,o,lane.2⟩
  outputMask:=by
    intro o _ j
    change Fin 3 at o
    let lane:=outputIndex fp8 p order o j
    exact Winner p ⟨lane.1,o,lane.2⟩
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>order.map (job a native fp8 p epsilon theta)
  encode:=id
  observe:=fun _ _=>id

/-- Only selected INT8 weight/RMS arrays require valid signed bytes. The FP16
branch has no integer condition; unrelated memory is never constrained. -/
def InputDomain (fp8 : Bool) (p : Fin 3 → Arguments) (vals : ArrayValues α (inputs fp8 p)) : Prop :=
  match fp8,vals with
  | Bool.false,_=>True
  | Bool.true,vals=>∀ phase off,
      Float8E5Bits.isInt8 (wValues Bool.true p vals phase off)=Bool.true ∧
      Float8E5Bits.isInt8 (rmsValues Bool.true p vals phase off)=Bool.true

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IO

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Target
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

/-- Logical arrays decoded at the source's explicit precision. E5 RMS and
weight conversions deliberately retain their different conversion paths. -/
def rowInput {α : Type} (a : Arithmetic α) (fp8 : Bool) (p : Fin 3 → Arguments)
    (vals : ArrayValues α (IO.inputs fp8 p)) (phase : Fin 3) (batch row : Nat) :
    RmsMatmulRbe.RowInput α := by
  cases fp8
  · exact {
      x:=fun k=>IO.xValues Bool.false p vals phase
        (batch*(p phase).sxb+row*(p phase).sxm+k*(p phase).sxk)
      rms:=fun k=>kernelCast a .fp16 .real (IO.rmsValues Bool.false p vals phase (k*(p phase).srms))
      weight:=fun n k=>kernelCast a .fp16 .real
        (IO.wValues Bool.false p vals phase (k*(p phase).swk+n*(p phase).swn)) }
  · exact {
      x:=fun k=>IO.xValues Bool.true p vals phase
        (batch*(p phase).sxb+row*(p phase).sxm+k*(p phase).sxk)
      rms:=fun k=>kernelCast a .fp16 .real (kernelCast a .f8e5 .fp16
        (a.e5Bits .f8e5 (Float8E5Bits.ofInt (IO.rmsValues Bool.true p vals phase (k*(p phase).srms)))))
      weight:=fun n k=>kernelCast a .fp16 .real (kernelCast a .real .fp16 (kernelCast a .f8e5 .real
        (a.e5Bits .f8e5 (Float8E5Bits.ofInt (IO.wValues Bool.true p vals phase
          (k*(p phase).swk+n*(p phase).swn)))))) }


end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Target

/-! Named mathematical X/W/RMS arrays for every Q/K/V phase. The graph keeps
all source backing words as well: arbitrary strides, aliases and unused cells
are retained rather than projected away. Outputs retain every scheduled lane;
the original judgment still observes exactly the final winning writes. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Logical
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

@[ext] structure Backing (fp8 : Bool) where
  x : Fin 3 → Nat → ℝ
  weight : Fin 3 → Nat → Value ℝ (IO.weightDType fp8)
  rms : Fin 3 → Nat → Value ℝ (IO.weightDType fp8)

/-- All nine original arrays, including their unobserved suffixes. -/
def backingEquiv (fp8 : Bool) (p : Fin 3 → Arguments) :
    ArrayValues ℝ (IO.inputs fp8 p) ≃ Backing fp8 where
  toFun := fun vals => ⟨IO.xValues fp8 p vals, IO.wValues fp8 p vals, IO.rmsValues fp8 p vals⟩
  invFun := by
    intro vals
    refine Fin.cases (vals.x 0) ?_
    refine Fin.cases (vals.x 1) ?_
    refine Fin.cases (vals.x 2) ?_
    refine Fin.cases (vals.weight 0) ?_
    refine Fin.cases (vals.weight 1) ?_
    refine Fin.cases (vals.weight 2) ?_
    refine Fin.cases (vals.rms 0) ?_
    refine Fin.cases (vals.rms 1) ?_
    exact Fin.cases (vals.rms 2) (fun i => i.elim0)
  left_inv := by
    intro vals
    funext i off
    fin_cases i <;> rfl
  right_inv := by
    intro vals
    apply Backing.ext <;> funext phase off <;> fin_cases phase <;> rfl

structure Arrays where
  x : Fin 3 → Nat → Nat → Nat → ℝ
  weight : Fin 3 → Nat → Nat → ℝ
  rms : Fin 3 → Nat → ℝ

/-- Decode the actual source precision and strides once at the IO boundary. -/
def arraysOfBacking (fp8 : Bool) (p : Fin 3 → Arguments) (backing : Backing fp8) : Arrays where
  x := fun phase batch row k =>
    (Target.rowInput mathArithmetic fp8 p ((backingEquiv fp8 p).symm backing) phase batch row).x k
  weight := fun phase column k =>
    (Target.rowInput mathArithmetic fp8 p ((backingEquiv fp8 p).symm backing) phase 0 0).weight column k
  rms := fun phase k =>
    (Target.rowInput mathArithmetic fp8 p ((backingEquiv fp8 p).symm backing) phase 0 0).rms k

structure Inputs (fp8 : Bool) (p : Fin 3 → Arguments) where
  arrays : Arrays
  backing : Backing fp8
  connected : arrays = arraysOfBacking fp8 p backing

/-- The connection graph is bijective; logical arrays are linked views of the
original storage, not arbitrary independent values when source strides alias. -/
def inputEquiv (fp8 : Bool) (p : Fin 3 → Arguments) :
    ArrayValues ℝ (IO.inputs fp8 p) ≃ Inputs fp8 p where
  toFun := fun raw => ⟨arraysOfBacking fp8 p (backingEquiv fp8 p raw), backingEquiv fp8 p raw, rfl⟩
  invFun := fun xs => (backingEquiv fp8 p).symm xs.backing
  left_inv := fun raw => (backingEquiv fp8 p).symm_apply_apply raw
  right_inv := by
    rintro ⟨arrays, backing, connected⟩
    cases connected
    simp only [Equiv.apply_symm_apply]

/-- The selected INT8 branch admits exactly signed-byte weight and RMS words. -/
def Domain (fp8 : Bool) {p : Fin 3 → Arguments} (xs : Inputs fp8 p) : Prop :=
  match fp8, xs with
  | Bool.false, _ => True
  | Bool.true, xs => ∀ phase off,
      Float8E5Bits.isInt8 (xs.backing.weight phase off) = Bool.true ∧
      Float8E5Bits.isInt8 (xs.backing.rms phase off) = Bool.true

abbrev Outputs (order : List (Nat → Nat)) :=
  Fin 3 → Fin order.length → Fin 16 → Fin 64 → ℝ

/-- The Real carrier of each actual FP16/E5 output has the same tensor axes. -/
def phaseOutputEquiv (fp8 : Bool) (p : Fin 3 → Arguments)
    (order : List (Nat → Nat)) (phase : Fin 3) :
    Values ℝ (IO.outputs fp8 p order)[phase].dtype (IO.outputs fp8 p order)[phase].shape ≃
      (Fin order.length → Fin 16 → Fin 64 → ℝ) := by
  rw [IO.output_channel]
  unfold IO.outputDType
  split <;> exact Values.tensor3Equiv ℝ _ order.length 16 64

/-- Phase/launch/row/column coordinates retain all typed output lanes. -/
def outputEquiv (fp8 : Bool) (p : Fin 3 → Arguments) (order : List (Nat → Nat)) :
    ChannelValues ℝ (IO.outputs fp8 p order) ≃ Outputs order :=
  Equiv.piCongrRight (fun phase => phaseOutputEquiv fp8 p order phase)

def qkvIO (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat)) :
    LogicalDomainIO (ArrayValues ℝ (IO.inputs fp8 p))
      (ChannelValues ℝ (IO.outputs fp8 p order)) (Inputs fp8 p) (Outputs order)
      (IO.qkvIO mathArithmetic Bool.false fp8 p epsilon.value theta.value order).ImplementsOn where
  inputs := inputEquiv fp8 p
  outputs := outputEquiv fp8 p order

/-- Mathematical row supplied to the normalized projection. -/
def rowInputs {fp8 : Bool} {p : Fin 3 → Arguments} (xs : Inputs fp8 p)
    (phase : Fin 3) (batch row : Nat) : RmsMatmulRbe.RowInput ℝ where
  x := xs.arrays.x phase batch row
  rms := xs.arrays.rms phase
  weight := xs.arrays.weight phase

def batch (order : List (Nat → Nat)) (launch : Fin order.length) : Nat := order[launch] 0
def row (p : Fin 3 → Arguments) (order : List (Nat → Nat)) (phase : Fin 3)
    (launch : Fin order.length) (rowLane : Fin 16) : Nat :=
  order[launch] 1 / columns (p phase) * 16 + rowLane.val
def column (p : Fin 3 → Arguments) (order : List (Nat → Nat)) (phase : Fin 3)
    (launch : Fin order.length) (columnLane : Fin 64) : Nat :=
  order[launch] 1 % columns (p phase) * 64 + columnLane.val

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Logical
