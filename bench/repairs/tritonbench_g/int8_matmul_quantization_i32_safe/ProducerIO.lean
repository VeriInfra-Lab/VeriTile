import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerTargets
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section

/-- The producer reads the original FP32 matrix and stores signed INT8 data
plus one FP16 scale per row. Structural uses .real for its FP32 source carrier
and .int for its exact signed integer cells. -/
@[reducible] def inputs (p : Geometry) : List IOChannel :=
  [⟨.real,[p.rows,p.columns],"x"⟩]
@[reducible] def outputs (p : Geometry) : List IOChannel :=
  [⟨.int,[p.rows,p.columns],"q"⟩,⟨.fp16,[p.rows],"scales"⟩]
abbrev Inputs (α : Type) (p : Geometry) := ChannelValues α (inputs p)
abbrev Outputs (α : Type) (p : Geometry) := ChannelValues α (outputs p)

def layout (α : Type) (p : Geometry) : MemoryIO α where
  inputs:=inputs p
  outputs:=outputs p
  scratch:=[]
  buffers:=["x","q","scales"]
  read:=fun port _=>match port with
    | ⟨0,_⟩=>fun j=>j.1.val*p.columns+j.2.1.val
  inputMask:=fun _ _ _=>True
  write:=fun port _=>match port with
    | ⟨0,_⟩=>fun j=>j.1.val*p.columns+j.2.1.val
    | ⟨1,_⟩=>fun j=>j.1.val
  outputMask:=fun _ _ _=>True
  scratchWrite:=fun port=>port.elim0
  scratchMask:=fun port=>port.elim0

/-- One actual complete source row invocation. Other PID axes and local
registers retain the original launch behavior; every row executes both loops. -/
def invocation {α : Type} (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Geometry) (r : Fin p.rows) : Invocation α where
  kernel:=Source.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p))
  backend:=KernelProducer.backend M integer8
  pids:=fun _=>r.val
  setPids:=fun ax=>decide (ax=0)

def io {α : Type} (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Geometry) (order : I32Safe.Producer.Schedule p) : ProgramIO α α where
  toMemoryIO:=layout α p
  schedule:=fun _=>order.order.map (invocation M integer8 p)
  encode:=id
  observe:=fun _ _=>id

/-- Finite logical matrix extended only for the independent scalar formulas.
The formulas' masked accesses never depend on this outside filler. -/
def values {α : Type} (zero : α) (p : Geometry) (xs : Inputs α p) : Nat→Nat→α :=
  fun r c=>if hr:r<p.rows then if hc:c<p.columns then xs 0 (⟨r,hr⟩,⟨c,hc⟩,()) else zero else zero

def mathematical (p : Geometry) (xs : Inputs ℝ p) : Outputs ℝ p :=
  fun port=>match port with
  | ⟨0,_⟩=>fun j=>(KernelProducerTargets.mathematicalByte (values 0 p xs j.1.val) p.columns j.2.1.val).toInt
  | ⟨1,_⟩=>fun j=>KernelProducerTargets.mathematicalScale (values 0 p xs j.1.val) p.columns

def nativeFormula (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : I32Safe.Producer.Selection p) (xs : Inputs Word32 p) : Outputs Word32 p :=
  fun port=>match port with
  | ⟨0,_⟩=>fun j=>(KernelProducerTargets.nativeByte f p selection
      (values (FP.Value.zero .fp32) p xs j.1.val) j.2.1.val).toInt
  | ⟨1,_⟩=>fun j=>KernelScalarProfile.pack16 (KernelProducerTargets.nativeStoredScale f p selection
      (values (FP.Value.zero .fp32) p xs j.1.val))

def realIO (p : Geometry) (selection : I32Safe.Producer.Selection p)
    (rest : Algebra ℝ) (order : I32Safe.Producer.Schedule p) : ProgramIO ℝ ℝ :=
  io (KernelScalarProfile.mathematical selection.tree rest) realInteger8 p order

def nativeIO (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : I32Safe.Producer.Selection p) (rest : Algebra Word32)
    (order : I32Safe.Producer.Schedule p) : ProgramIO Word32 Word32 :=
  io (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order

#specNonCircular mathematical avoiding [FP.Structural.exec,run,KernelProducerGrid.run,KernelProducer.execution]
#specNonCircular nativeFormula avoiding [FP.Structural.exec,run,KernelProducerGrid.run,KernelProducer.execution]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
structure LogicalOutputs (p : Geometry) where
  codes : Fin p.rows → Fin p.columns → Int
  scales : Fin p.rows → ℝ

def logicalInputs (p : Geometry) : Inputs ℝ p ≃ (Fin p.rows → Fin p.columns → ℝ) where
  toFun xs := fun r c => xs 0 (r,c,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def logicalOutputs (p : Geometry) : Outputs ℝ p ≃ LogicalOutputs p where
  toFun xs := ⟨fun r c => xs 0 (r,c,()),fun r => xs 1 (r,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.codes j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.scales j.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def extendRow {p : Geometry} (xs : Fin p.rows → Fin p.columns → ℝ) (r : Fin p.rows) (c : Nat) : ℝ :=
  if hc : c<p.columns then xs r ⟨c,hc⟩ else 0
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
noncomputable section
open KernelProducerGrid (Geometry)
def logicalIO (p : Geometry) (selection : I32Safe.Producer.Selection p) (rest : Algebra ℝ) (order : I32Safe.Producer.Schedule p) : LogicalProgramIO ℝ ℝ (Fin p.rows → Fin p.columns → ℝ) (LogicalOutputs p) where
  core := realIO p selection rest order
  inputs := logicalInputs p
  outputs := logicalOutputs p
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
