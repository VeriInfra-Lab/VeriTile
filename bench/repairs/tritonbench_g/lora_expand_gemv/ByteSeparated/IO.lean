import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.SourceCompatibility
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native
import VeriTile.Triton.Float.ObservedInputCellsIO
import VeriTile.Triton.Float.LogicalProgramIO


/-! Public IO for the selected byte-separated source. Floating source values
are the original address-labeled graph; mathematical/native payloads are the
separate `data`/`bits` interpretations of those same labels. Exact input cells
retain the unchanged Y result when an index is the sentinel -1. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.LoraIO
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

def inputs (g : Geometry) : List IOChannel :=
  [⟨.int,[g.B],"I"⟩,⟨.real,[g.B,g.K],"X"⟩,
    ⟨.fp16,[g.L,g.N,g.K],"W"⟩,⟨.real,[g.B,g.N],"Y"⟩]
def outputs (g : Geometry) : List IOChannel := [⟨.real,[g.B,g.N],"Y"⟩]

def memoryIO (g : Geometry) : MemoryIO Word where
  inputs := inputs g
  outputs := outputs g
  buffers := ["I","X","W","Y"]
  read := Fin.cases (fun _ j => j.1.val) (Fin.cases
    (fun _ j => xoff g j.1.val j.2.1.val) (Fin.cases
    (fun _ j => woff g j.1.val j.2.1.val j.2.2.1.val)
    (Fin.cases (fun _ j => yoff g j.1.val j.2.1.val) (fun i => Fin.elim0 i))))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => yoff g j.1.val j.2.1.val) (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratchWrite := by intro i; exact Fin.elim0 i
  scratchMask := by intro i; exact Fin.elim0 i

def launch (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (q : Nat) : Invocation Word where
  kernel := source .singleHalf .single g.parameters
  backend := Branches.engine rest .ieeeValue plans
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then q%g.N else if ax=1 then q/g.N else 0
  numPids := some (fun ax => if ax=0 then g.N else if ax=1 then g.B else 1)

def io (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (order : List Nat) : ProgramIO Word Word where
  toMemoryIO := memoryIO g
  schedule := fun _ => order.map (launch rest plans g)
  encode := id
  observe := fun _ _ => id

/-- The historical graph input domain is explicit. Encoding is identity on
these logical labels; neither the source nor the encoder sees `data`/`bits`. -/
def labeledInputs (g : Geometry) (indices : Nat → Int) : ChannelValues Word (inputs g) :=
  Fin.cases (fun j => logicalIndex g indices j.1.val) (Fin.cases
    (fun j => .leaf "X" (xoff g j.1.val j.2.1.val)) (Fin.cases
    (fun j => .leaf "W" (woff g j.1.val j.2.1.val j.2.2.1.val))
    (Fin.cases (fun j => .leaf "Y" (yoff g j.1.val j.2.1.val)) (fun i => Fin.elim0 i))))
def domain (g : Geometry) (indices : Nat → Int) (vals : ChannelValues Word (inputs g)) : Prop :=
  vals=labeledInputs g indices

def graphResult (plans : ∀ n, ReductionPlan n) (g : Geometry) (indices : Nat → Int)
    (b n : Nat) : Word :=
  if logicalIndex g indices b=-1 then .leaf "Y" (yoff g b n)
  else value plans g indices b n

def graphTarget (plans : ∀ n, ReductionPlan n) (g : Geometry) (indices : Nat → Int)
    (_ : ChannelValues Word (inputs g)) : ChannelValues Word (outputs g) :=
  Fin.cases (fun j => graphResult plans g indices j.1.val j.2.1.val) (fun i => Fin.elim0 i)

def realObservation (data : RegionName → Nat → ℝ) (g : Geometry) :
    (o : Fin (outputs g).length) → TileIndex (outputs g)[o].shape → Word → ℝ :=
  fun _ _ => Real.eval data

def realTarget (data : RegionName → Nat → ℝ) (g : Geometry) (indices : Nat → Int)
    (_ : ChannelValues Word (inputs g)) : ChannelValues ℝ (outputs g) :=
  Fin.cases (fun j => Real.target data g indices j.1.val j.2.1.val) (fun i => Fin.elim0 i)

def nativeObservation (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (bits : RegionName → Nat → Native.Bits) (g : Geometry) :
    (o : Fin (outputs g).length) → TileIndex (outputs g)[o].shape → Word → Native.Bits :=
  fun _ _ => Native.eval ops schedule bits

def nativeTarget (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (indices : Nat → Int)
    (bits : RegionName → Nat → Native.Bits) (_ : ChannelValues Word (inputs g)) :
    ChannelValues Native.Bits (outputs g) :=
  Fin.cases (fun j => Native.eval ops schedule bits (graphResult plans g indices j.1.val j.2.1.val)) (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.LoraIO

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.ReaderIO
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

/-- The exact graph-label inputs, including the prior Y buffer. -/
structure Inputs (g : Geometry) where
  index : Fin g.B → Int
  X : Fin g.B → Fin g.K → Word
  W : Fin g.L → Fin g.N → Fin g.K → Word
  Y : Fin g.B → Fin g.N → Word

def inputs (g : Geometry) : ChannelValues Word (LoraIO.inputs g) ≃ Inputs g where
  toFun := fun xs => ⟨(fun j => xs (0 : Fin 4) (j,())),
    (fun j k => xs (1 : Fin 4) (j,k,())),
    (fun l n k => xs (2 : Fin 4) (l,n,k,())), (fun j n => xs (3 : Fin 4) (j,n,()))⟩
  invFun := fun xs => Fin.cases (fun j => xs.index j.1) (Fin.cases
    (fun j => xs.X j.1 j.2.1) (Fin.cases (fun j => xs.W j.1 j.2.1 j.2.2.1)
    (Fin.cases (fun j => xs.Y j.1 j.2.1) (fun i => i.elim0))))
  left_inv := by
    intro xs; funext i j; fin_cases i
    · rcases j with ⟨a,⟨⟩⟩; rfl
    · rcases j with ⟨a,b,⟨⟩⟩; rfl
    · rcases j with ⟨a,b,c,⟨⟩⟩; rfl
    · rcases j with ⟨a,b,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

/-- Logical real payloads interpreted from the same address labels. This is
observer data, separate from the typed graph inputs read by the actual source. -/
structure RealData where
  X : Nat → Nat → ℝ
  W : Nat → Nat → Nat → ℝ
  Y : Nat → Nat → ℝ

def observed (data : RegionName → Nat → ℝ) (g : Geometry) : RealData where
  X := fun b k => data "X" (xoff g b k)
  W := fun l n k => data "W" (woff g l n k)
  Y := fun b n => data "Y" (yoff g b n)

def labeled (g : Geometry) (indices : Nat → Int) (xs : Inputs g) : Prop :=
  (inputs g).symm xs = LoraIO.labeledInputs g indices

def view (rest : Algebra Word) (plans : ∀ n,ReductionPlan n) (g : Geometry)
    (data : RegionName → Nat → ℝ) (order : List Nat) :
    LogicalDomainIO (ChannelValues Word (LoraIO.inputs g)) (ChannelValues ℝ (LoraIO.outputs g))
      (Inputs g) (Fin g.B → Fin g.N → ℝ)
      ((LoraIO.io rest plans g order).ObservesOnCells (LoraIO.realObservation data g)) where
  inputs := inputs g
  outputs := (ChannelValues.singleEquiv ℝ _).trans (Values.matrixEquiv ℝ .real g.B g.N)
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.ReaderIO
