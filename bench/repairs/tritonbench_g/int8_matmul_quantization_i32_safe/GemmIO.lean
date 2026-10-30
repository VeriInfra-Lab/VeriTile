import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GridResults
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.IntegerDotBridge
import VeriTile.Triton.Float.IntegerDotPerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
open IntScaledMatmul.Revision001.Geometry
noncomputable section

/-- Structural .real is the original FP32 channel. Half and single storage
remain distinct; signed INT8 cells use exact signed integer payloads. -/
@[reducible] def dtype : Precision→TileDType | .half=>.fp16 | .single=>.real
@[reducible] def value (kind : Precision) (x : F) : Value F (dtype kind) :=
  match kind with | .half=>x | .single=>x
@[reducible] def fromValue (kind : Precision) (x : Value F (dtype kind)) : F :=
  match kind with | .half=>x | .single=>x

@[reducible] def inputs (s : Shape) (rowScale : Precision) : List IOChannel :=
  [⟨.int,[s.M,s.K],"A"⟩,⟨.int,[s.K,s.N],"B"⟩,
   ⟨dtype rowScale,[s.M],"AS"⟩,⟨.real,[s.N],"BS"⟩]
@[reducible] def outputs (s : Shape) (output : Precision) : List IOChannel :=
  [⟨dtype output,[s.M,s.N],"C"⟩]
abbrev Inputs (F : Type) (s : Shape) (rowScale : Precision) := ChannelValues F (inputs s rowScale)
abbrev Outputs (F : Type) (s : Shape) (output : Precision) := ChannelValues F (outputs s output)

/-- The original canonical row-major A, selected row/column-major B, row and
column scales, and contiguous C. C is reset by fixed typed zero data, not a
caller-supplied initial-output premise. -/
@[reducible] def layout (F : Type) (s : Shape) (rowScale output : Precision) (columnMajor : Bool) : MemoryIO F where
  inputs:=inputs s rowScale
  outputs:=outputs s output
  buffers:=["A","B","AS","BS","C"]
  scratch:=[]
  read:=fun port _=>match port with
    | ⟨0,_⟩=>fun j=>j.1.val*max s.K 1+j.2.1.val
    | ⟨1,_⟩=>fun j=>if columnMajor then j.2.1.val*max s.K 1+j.1.val else j.1.val*max s.N 1+j.2.1.val
    | ⟨2,_⟩=>fun j=>j.1.val
    | ⟨3,_⟩=>fun j=>j.1.val
  inputMask:=fun _ _ _=>True
  write:=fun port _=>match port with
    | ⟨0,_⟩=>fun j=>j.1.val*s.N+j.2.1.val
  outputMask:=fun _ _ _=>True
  scratchWrite:=fun port=>port.elim0
  scratchMask:=fun port=>port.elim0

/-- Every grouped CTA and split is retained, including padded lanes. -/
abbrev Program (s : Shape) (c : Config) := Fin (count (geometry s c))×Fin c.S

def programs (s : Shape) (c : Config) : List (Program s c) :=
  (List.finRange (count (geometry s c))).product (List.finRange c.S)

def owner (s : Shape) (c : Config) (cell : MatrixCell s) : Fin (count (geometry s c)) :=
  ⟨Classical.choose (Grouped.every_cell s c cell),(Classical.choose_spec (Grouped.every_cell s c cell)).1⟩

def events (s : Shape) (c : Config) (cell : MatrixCell s) : List (Program s c) :=
  (List.finRange c.S).map (fun split=>(owner s c cell,split))

/-- The original arbitrary single-cell interleaving determines each cell's
split order. No serial whole-CTA schedule replaces that interleaving. -/
def order (s : Shape) (c : Config) (q : Schedule s c) (cell : MatrixCell s) : List (Program s c) :=
  (projection cell q.events).map (fun split=>(owner s c cell,split))

def invocation (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : Program s c) : IntegerDotInvocation F where
  kernel:=I32Safe.Gemm.source rowScale output (I32Safe.Gemm.parameters s c columnMajor)
  backend:=M
  arithmetic:=SharedInteger.arithmetic carrier ops zero plans
  context:=SharedInteger.context plans
  pids:=fun ax=>if ax=0 then p.1.val else p.2.val
  setPids:=fun ax=>decide (ax=0 ∨ ax=1)

@[reducible] def io (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    IntegerDotPerAddressProgramIO F F F (Program s c) where
  toMemoryIO:=layout F s rowScale output columnMajor
  programs:=fun _=>programs s c
  invocation:=fun _=>invocation M carrier ops zero plans rowScale output s c columnMajor
  events:=fun _ port=>match port with | ⟨0,_⟩=>fun j=>events s c (j.1,j.2.1)
  order:=fun _ port=>match port with | ⟨0,_⟩=>fun j=>order s c q (j.1,j.2.1)
  encode:=id
  observe:=fun _ _=>id

/-- Exact signed-INT8 input payload domain; this is a type representation
condition, not a floating numerical law or experimental assumption. -/
def SignedInputs (s : Shape) (rowScale : Precision) (xs : Inputs F s rowScale) : Prop :=
  (∀ j, (BitVec.ofInt 8 (xs 0 j)).toInt=xs 0 j) ∧
  (∀ j, (BitVec.ofInt 8 (xs 1 j)).toInt=xs 1 j)

/-- Finite tensor values extended only for the independent scalar formulas. -/
def input (zero : F) (s : Shape) (rowScale : Precision) (xs : Inputs F s rowScale) : I32Safe.Gemm.Input F where
  A:=fun r k=>if hr:r<s.M then if hk:k<s.K then BitVec.ofInt 8 (xs 0 (⟨r,hr⟩,⟨k,hk⟩,())) else 0 else 0
  B:=fun k n=>if hk:k<s.K then if hn:n<s.N then BitVec.ofInt 8 (xs 1 (⟨k,hk⟩,⟨n,hn⟩,())) else 0 else 0
  aScale:=fun r=>if hr:r<s.M then fromValue rowScale (xs 2 (⟨r,hr⟩,())) else zero
  bScale:=fun n=>if hn:n<s.N then xs 3 (⟨n,hn⟩,()) else zero

/-- Independent dense scaled integer matrix product. -/
def mathematical (s : Shape) (rowScale output : Precision) (xs : Inputs ℝ s rowScale) : Outputs ℝ s output :=
  fun port=>match port with
  | ⟨0,_⟩=>fun j=>value output
      (target s (I32Safe.Gemm.realInput s (input 0 s rowScale xs)) j.1 j.2.1)

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

/-- Independent ordered native scalar graph, including each original cast,
mandatory half partial and the selected per-cell atomic order. -/
def nativeFormula (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) (xs : Inputs Word32 s rowScale) : Outputs Word32 s output :=
  fun port=>match port with | ⟨0,_⟩=>fun j=>
    (Grid.nativeWord f tree rest plans rowScale output s c columnMajor
      (input (FP.Value.zero .fp32) s rowScale xs) q (j.1,j.2.1)).read (dtype output)

def realIO (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :=
  io (KernelScalarProfile.mathematical tree rest) mathematicalIntegers KernelIntegerIdeal.operations.toOps
    rfl plans rowScale output s c columnMajor q

def nativeIO (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :=
  io (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps
    rfl plans rowScale output s c columnMajor q

def initialValues (zero : F) (s : Shape) (output : Precision) :
    (key : (o : Fin (outputs s output).length)×TileIndex (outputs s output)[o].shape)→
      Value F (outputs s output)[key.1].dtype :=
  fun ⟨port,_⟩=>match port with | ⟨0,_⟩=>value output zero

#specNonCircular mathematical avoiding [KernelExecution.exec,Grid.run,Grid.launch,FP.Structural.exec]
#specNonCircular nativeFormula avoiding [KernelExecution.exec,Grid.run,Grid.launch,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
structure LogicalInputs (s : Shape) where
  a : Fin s.M → Fin s.K → Int
  b : Fin s.K → Fin s.N → Int
  rowScale : Fin s.M → ℝ
  columnScale : Fin s.N → ℝ

def logicalInputs (s : Shape) (rowScale : Precision) : Inputs ℝ s rowScale ≃ LogicalInputs s where
  toFun xs := ⟨fun r k => xs 0 (r,k,()),fun k c => xs 1 (k,c,()),
    fun r => fromValue rowScale (xs 2 (r,())),fun c => xs 3 (c,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.a j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.b j.1 j.2.1
    | ⟨2,_⟩ => fun j => value rowScale (xs.rowScale j.1)
    | ⟨3,_⟩ => fun j => xs.columnScale j.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; cases rowScale <;> rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; cases rowScale <;> rfl

def logicalOutputs (s : Shape) (output : Precision) : Outputs ℝ s output ≃ (Fin s.M → Fin s.N → ℝ) where
  toFun xs := fun r c => fromValue output (xs 0 (r,c,()))
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => value output (xs j.1 j.2.1)
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; cases output <;> rfl
  right_inv := by intro xs; cases output <;> rfl
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
def logicalIO (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (initial : (realIO tree rest plans rowScale output s c columnMajor q).RawOutputs) :
    LogicalDomainIO (Inputs ℝ s rowScale) (Outputs ℝ s output) (LogicalInputs s) (Fin s.M → Fin s.N → ℝ)
      (fun domain f => (realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells domain initial f) where
  inputs := logicalInputs s rowScale
  outputs := logicalOutputs s output

def logicalSignedInputs (s : Shape) (rowScale : Precision) (xs : LogicalInputs s) : Prop :=
  SignedInputs s rowScale ((logicalInputs s rowScale).symm xs)
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
