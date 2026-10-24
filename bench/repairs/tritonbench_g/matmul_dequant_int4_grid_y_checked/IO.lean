import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Kernels
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.DSL
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Grid
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProgramEquivalence
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Spec
import VeriTile.Triton.Float.LogicalProgramIO


namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

def dtype : Source.Profile → TileDType | .half => .fp16 | .single => .fp32

def inputs (f : Source.Profile) (p : Source.Parameters) : List IOChannel :=
  [⟨.int,[p.rows,p.columns],p.weight⟩,⟨.int,[p.rows,p.columns],p.zero⟩,
   ⟨dtype f,[p.rows,p.columns],p.scale⟩]

def outputs (f : Source.Profile) (p : Source.Parameters) : List IOChannel :=
  [⟨dtype f,[p.rows,p.columns],p.output⟩]

abbrev Inputs (α : Type) (f : Source.Profile) (p : Source.Parameters) := ChannelValues α (inputs f p)

def invocation {α : Type} (f : Source.Profile) (M : Algebra α)
    (p : Source.Parameters) (pid : Nat) : Invocation α where
  kernel := Source.source f p
  backend := M
  pids := fun ax => if ax=0 then pid/(MatmulDequantInt4.ceil p.columns p.blockColumns)
    else if ax=1 then pid%(MatmulDequantInt4.ceil p.columns p.blockColumns) else 0

/-- Public logical read windows name the packed word and scale needed at each
output coordinate. Shared packed/grouped words intentionally repeat addresses.
The exact signed INT32 source load, chosen FP16/FP32 profile and all strides
remain explicit; no input-value condition or source-success premise is hidden. -/
@[reducible] def dequantIO {α : Type} (f : Source.Profile) (M : Algebra α)
    (p : Source.Parameters) : ProgramIO α α where
  inputs := inputs f p
  outputs := outputs f p
  scratch := []
  buffers := [p.weight,p.zero,p.scale,p.output]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.1.val/8*p.weightRowStride+j.2.1.val*p.weightColumnStride
    | ⟨1,_⟩ => fun j => j.1.val/p.groupSize*p.zeroRowStride+j.2.1.val/8*p.zeroColumnStride
    | ⟨2,_⟩ => fun j => j.1.val/p.groupSize*p.scaleRowStride+j.2.1.val*p.scaleColumnStride
    | ⟨k+3,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.1.val*p.columns+j.2.1.val
    | ⟨k+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range (KernelGrid.count p)).map (invocation f M p)
  encode := id
  observe := fun _ _ => id

/-- Same complete two-axis grid and public memory ABI, with the concrete
unit-column-stride source specialization. -/
def unitInvocation {α : Type} (f : Source.Profile) (M : Algebra α)
    (p : Source.Parameters) (pid : Nat) : Invocation α :=
  { invocation f M p pid with kernel := unitSource f p }

def unit {α : Type} (f : Source.Profile) (M : Algebra α) (p : Source.Parameters) :
    ProgramIO.Variant (dequantIO f M p) where
  schedule := fun _ => (List.range (KernelGrid.count p)).map (unitInvocation f M p)
  observe := fun _ _ => id

def packed {α : Type} {f : Source.Profile} {p : Source.Parameters}
    (xs : Inputs α f p) (zero : Bool) (x : KernelGrid.Coordinate p) : Int :=
  if zero then xs ⟨1,by simp [inputs]⟩ (x.1,x.2,PUnit.unit)
  else xs ⟨0,by simp [inputs]⟩ (x.1,x.2,PUnit.unit)

def scale {α : Type} (f : Source.Profile) {p : Source.Parameters}
    (xs : Inputs α f p) (x : KernelGrid.Coordinate p) : α := by
  cases f
  · exact xs ⟨2,by simp [inputs]⟩ (x.1,x.2,PUnit.unit)
  · exact xs ⟨2,by simp [inputs]⟩ (x.1,x.2,PUnit.unit)

/-- Independent signed packed-nibble difference; the logical operands are
input words, not memory reads or source execution results. -/
def difference {α : Type} {f : Source.Profile} {p : Source.Parameters}
    (xs : Inputs α f p) (x : KernelGrid.Coordinate p) : Int :=
  Source.nibble (packed xs Bool.false x) x.1.val - Source.nibble (packed xs Bool.true x) x.2.val

def result {α : Type} (f : Source.Profile) (p : Source.Parameters)
    (fn : KernelGrid.Coordinate p → α) : ChannelValues α (outputs f p) := by
  cases f
  · exact Fin.cases (fun j => fn (j.1,j.2.1)) (fun i => i.elim0)
  · exact Fin.cases (fun j => fn (j.1,j.2.1)) (fun i => i.elim0)

#specNonCircular difference avoiding [Source.source,Source.execution,KernelGrid.result,dequantIO]
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.ReaderIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Packed words and scale at every logical output coordinate. The original
IO retains the shared packed/grouped addresses and selected source precision. -/
structure Inputs (p : Source.Parameters) where
  weight : Fin p.rows → Fin p.columns → Int
  zero : Fin p.rows → Fin p.columns → Int
  scale : Fin p.rows → Fin p.columns → ℝ

def inputs (f : Source.Profile) (p : Source.Parameters) : PublicIO.Inputs ℝ f p ≃ Inputs p where
  toFun := fun xs => ⟨(fun r c => PublicIO.packed xs Bool.false (r,c)),
    (fun r c => PublicIO.packed xs Bool.true (r,c)), (fun r c => PublicIO.scale f xs (r,c))⟩
  invFun := by
    intro xs; cases f
    · exact Fin.cases (fun j => xs.weight j.1 j.2.1)
        (Fin.cases (fun j => xs.zero j.1 j.2.1) (Fin.cases (fun j => xs.scale j.1 j.2.1) (fun i => i.elim0)))
    · exact Fin.cases (fun j => xs.weight j.1 j.2.1)
        (Fin.cases (fun j => xs.zero j.1 j.2.1) (Fin.cases (fun j => xs.scale j.1 j.2.1) (fun i => i.elim0)))
  left_inv := by
    cases f <;> intro xs <;> funext i j <;> fin_cases i <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by cases f <;> intro xs <;> cases xs <;> rfl

def outputs (f : Source.Profile) (p : Source.Parameters) :
    ChannelValues ℝ (PublicIO.outputs f p) ≃ (Fin p.rows → Fin p.columns → ℝ) := by
  cases f
  · exact (ChannelValues.singleEquiv ℝ _).trans (Values.matrixEquiv ℝ .fp16 p.rows p.columns)
  · exact (ChannelValues.singleEquiv ℝ _).trans (Values.matrixEquiv ℝ .fp32 p.rows p.columns)

def view (f : Source.Profile) (p : Source.Parameters) :
    LogicalProgramIO ℝ ℝ (Inputs p) (Fin p.rows → Fin p.columns → ℝ) where
  core := PublicIO.dequantIO f Spec.real p
  inputs := inputs f p
  outputs := outputs f p
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.ReaderIO
