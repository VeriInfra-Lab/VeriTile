import bench.tritonbench_g.int8_dequant_matmul.Mathematical
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteModel
import VeriTile.Triton.Float.MemoryIOJudgments
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Coverage
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IntegerDotBridge
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteStages
import VeriTile.Triton.Float.IntegerDotPerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOLayout
open VeriTile Triton FP.Structural
noncomputable section

/-- The actual six source regions. Integer channels retain the source's
explicit byte conversion; floating payloads retain their FP32/FP16 summands. -/
@[reducible] def inputs (s : Shape) : List IOChannel :=
  [⟨.int,[s.M,s.K],"A"⟩,⟨.int,[s.K,s.N],"B"⟩,
   ⟨.real,[s.M],"state_x"⟩,⟨.real,[s.N],"state_w"⟩,
   ⟨.real,[s.N],"bias"⟩,⟨.fp16,[s.M,s.N],"C"⟩]

@[reducible] def outputs (s : Shape) : List IOChannel := [⟨.fp16,[s.M,s.N],"C"⟩]

/-- C is an in-place input only for split K, where the existing initialization
hook provides FP16 zero. Physical readonly aliases are retained by the exact
sourceMemory binding in the physical bridge, without renaming the source. -/
@[reducible] def memory (F H : Type) (s : Shape) (c : Config) (hasBias : Bool) : MemoryIO (Sum F H) where
  inputs := inputs s
  outputs := outputs s
  scratch := []
  buffers := ["A","B","state_x","state_w","bias","C"]
  read := fun i _ j => TileShape.linearIndex (inputs s)[i].shape j
  write := fun o _ j => TileShape.linearIndex (outputs s)[o].shape j
  scratchWrite := fun i => Fin.elim0 i
  inputMask := fun i _ _ => (i.val=4 → hasBias=Bool.true) ∧ (i.val=5 → c.S≠1)
  outputMask := fun _ _ _ => True
  scratchMask := fun i => Fin.elim0 i

/-- The source's actual byte operands and FP32 scale/bias payloads. An erased
carrier with the wrong summand follows the old asF default behavior exactly. -/
def data (ops : Ops I F H) (s : Shape) (vals : ChannelValues (Sum F H) (inputs s)) : Mathematical.Inputs s F where
  a := fun r k => BitVec.ofInt 8 (vals 0 (r,k,PUnit.unit))
  b := fun k n => BitVec.ofInt 8 (vals 1 (k,n,PUnit.unit))
  rowScale := fun r => Quote.asF ops (vals 2 (r,PUnit.unit))
  columnScale := fun n => Quote.asF ops (vals 3 (n,PUnit.unit))
  bias := fun n => Quote.asF ops (vals 4 (n,PUnit.unit))

/-- Only the original split-K initialization requirement is imposed. -/
def domain (ops : Ops I F H) (s : Shape) (c : Config)
    (vals : ChannelValues (Sum F H) (inputs s)) : Prop :=
  c.S≠1 → ∀ (r : Fin s.M) (n : Fin s.N), vals 5 (r,n,PUnit.unit)=Sum.inr ops.hzero

def real (s : Shape) (hasBias : Bool) (vals : ChannelValues (Sum ℝ ℝ) (inputs s)) :
    ChannelValues ℝ (outputs s) :=
  Fin.cases (fun j => Mathematical.real s hasBias (data realOps s vals) (j.1,j.2.1))
    (fun i => Fin.elim0 i)

def native (profile : FixedISAProfile) (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (order : Schedule s c) (vals : ChannelValues (Sum Word32 Word16) (inputs s)) :
    ChannelValues Word16 (outputs s) :=
  Fin.cases (fun j => Mathematical.native profile s c selected hasBias order (data (fpOps profile) s vals) (j.1,j.2.1))
    (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOLayout

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOPrograms
open IntScaledMatmul.Revision001.Geometry
noncomputable section

/-- Every dispatched grouped tile and split, including padded lanes. -/
abbrev Program (s : Shape) (c : Config) := Fin (count (geometry s c)) × Fin c.S

def programs (s : Shape) (c : Config) : List (Program s c) :=
  (List.finRange (count (geometry s c))).product (List.finRange c.S)

/-- The unique grouped-grid tile that owns this output coordinate. Selection
depends only on source geometry, never on memory or a numerical result. -/
def owner (s : Shape) (c : Config) (cell : MatrixCell s) : Fin (count (geometry s c)) :=
  ⟨Classical.choose (Grouped.every_cell s c cell),
    (Classical.choose_spec (Grouped.every_cell s c cell)).1⟩

def events (s : Shape) (c : Config) (cell : MatrixCell s) : List (Program s c) :=
  (List.finRange c.S).map (fun split => (owner s c cell,split))

/-- The old physical-commit schedule determines a separate split order at
each address. It is deliberately not replaced by one global CTA order. -/
def order (s : Shape) (c : Config) (schedule : Schedule s c) (cell : MatrixCell s) : List (Program s c) :=
  (projection cell schedule.events).map (fun split => (owner s c cell,split))

def physicalProgram (p : PhysicalCommit s c) : Program s c := (p.pid,p.split)

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOPrograms

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO
open VeriTile Triton FP.Structural
noncomputable section

/-- The original grouped tile/split launch, source AST and scalar FP32 ABI.
Only ordinary launch registers and program IDs are installed. -/
def invocation (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c) :
    IntegerDotInvocation (Sum F H) where
  kernel := Quote.source s c hasBias
  backend := Quote.floating q
  arguments := [⟨"divfactor",.fp32,[],fun _=>.inl divisor⟩]
  pids := fun ax=>if ax=0 then p.1.val else if ax=1 then p.2.val else 0
  arithmetic := Quote.SharedInteger.arithmetic q
  context := Quote.SharedInteger.context q s.M s.N

/-- Complete grouped-grid execution with one independent source-derived
split order per output address. Six original regions, typed windows, optional
bias and in-place FP16 C are inherited from the public memory layout. -/
@[reducible] def programIO (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (schedule : Schedule s c) :
    IntegerDotPerAddressProgramIO (Sum F H) (Sum F H) H (IOPrograms.Program s c) where
  toMemoryIO := IOLayout.memory F H s c hasBias
  programs := fun _=>IOPrograms.programs s c
  invocation := fun _=>invocation q s c hasBias divisor
  events := fun _=>Fin.cases (fun j=>IOPrograms.events s c (j.1,j.2.1)) (fun i=>Fin.elim0 i)
  order := fun _=>Fin.cases (fun j=>IOPrograms.order s c schedule (j.1,j.2.1)) (fun i=>Fin.elim0 i)
  encode := id
  observe := fun _ _=>Quote.asH q.ops

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO
open VeriTile Triton FP.Structural
noncomputable section
/-- All six original typed inputs, including optional/initial channels. The
sum carrier distinguishes the source's single and half precision values. -/
structure LogicalInputs (s : Shape) where
  a : Fin s.M → Fin s.K → Int
  b : Fin s.K → Fin s.N → Int
  rowScale : Fin s.M → Sum ℝ ℝ
  columnScale : Fin s.N → Sum ℝ ℝ
  bias : Fin s.N → Sum ℝ ℝ
  initial : Fin s.M → Fin s.N → Sum ℝ ℝ

def logicalInputs (s : Shape) : ChannelValues (Sum ℝ ℝ) (IOLayout.inputs s) ≃ LogicalInputs s where
  toFun xs := ⟨fun r k => xs 0 (r,k,()),fun k c => xs 1 (k,c,()),
    fun r => xs 2 (r,()),fun c => xs 3 (c,()),fun c => xs 4 (c,()),fun r c => xs 5 (r,c,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.a j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.b j.1 j.2.1
    | ⟨2,_⟩ => fun j => xs.rowScale j.1
    | ⟨3,_⟩ => fun j => xs.columnScale j.1
    | ⟨4,_⟩ => fun j => xs.bias j.1
    | ⟨5,_⟩ => fun j => xs.initial j.1 j.2.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (s : Shape) : ChannelValues ℝ (IOLayout.outputs s) ≃ (Fin s.M → Fin s.N → ℝ) where
  toFun xs := fun r c => xs 0 (r,c,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def mathematicalInput (s : Shape) (xs : LogicalInputs s) : Mathematical.Inputs s ℝ :=
  IOLayout.data realOps s ((logicalInputs s).symm xs)
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO
open VeriTile Triton FP.Structural
noncomputable section
def logicalIO (s : Shape) (c : Config) (selected : Selection s c) (hasBias : Bool) (schedule : Schedule s c) :
    LogicalDomainIO (ChannelValues (Sum ℝ ℝ) (IOLayout.inputs s)) (ChannelValues ℝ (IOLayout.outputs s))
      (LogicalInputs s) (Fin s.M → Fin s.N → ℝ)
      (programIO (Quote.idealInteger s c selected) s c hasBias (1/16129) schedule).ImplementsOnCells where
  inputs := logicalInputs s
  outputs := logicalOutputs s

def logicalDomain (s : Shape) (c : Config) (xs : LogicalInputs s) : Prop :=
  IOLayout.domain Quote.integerRealOps s c ((logicalInputs s).symm xs)
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IO
