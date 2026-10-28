import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.PublicIO
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelNative
import VeriTile.Triton.Float.ObservedProgramIO

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Equivalence
open VeriTile.Triton FP.Structural Address
noncomputable section

/-- Actual FP32 source grid on arbitrary symbolic input terms. This preserves
the old input domain; native observation occurs only at the outputs. -/
abbrev layernormIO (R : NativeKernel.Reductions) (shape : Shape) (st : Strides)
    (epsilon : NativeKernel.Term) : ProgramIO NativeKernel.Term NativeKernel.Term where
  inputs := [⟨.fp32,[shape.sequences,shape.heads,shape.columns],"X"⟩,
    ⟨.fp32,[shape.sequences,shape.heads,shape.columns],"W"⟩]
  outputs := [⟨.fp32,[shape.sequences,shape.heads,shape.columns],"Y"⟩]
  scratch := []
  buffers := ["X","W","Y"]
  read := fun i => Fin.cases (fun _ j => xOffset st (j.1,j.2.1) j.2.2.1.val)
    (fun k => Fin.cases (fun _ j => wOffset st (j.1,j.2.1) j.2.2.1.val)
      (fun z => Fin.elim0 z) k) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => yOffset st (j.1,j.2.1) j.2.2.1.val)
    (fun z => Fin.elim0 z) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => (Source.programs shape).map (fun row => {
    kernel := Source.kernel (Source.parameters st shape.columns)
    backend := NativeKernel.algebra R
    arguments := [⟨"eps",.real,[],fun _ => epsilon⟩]
    pids := fun dim => if dim=0 then row.1.val else if dim=1 then row.2.val else 0})
  encode := id
  observe := fun _ _ => id

/-- Pure local native observation; it changes neither source execution nor
input binding. Scalar instructions, FMA/operand-order profile and complete
reduction trees remain the existing explicit model boundaries. -/
def observe (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (R : NativeKernel.Reductions) (shape : Shape) (st : Strides) (epsilon : NativeKernel.Term) :
    (o : Fin (layernormIO R shape st epsilon).outputs.length) →
      TileIndex ((layernormIO R shape st epsilon).outputs[o]).shape →
        NativeKernel.Term → NativeKernel.Word :=
  fun _ _ => NativeKernel.observe I P

/-- Independent native operation recurrence on the arbitrary logical terms. -/
def formula (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (R : NativeKernel.Reductions)
    (N : Nat) (x w : Nat → NativeKernel.Term) (epsilon : NativeKernel.Term) (j : Nat) :
    NativeKernel.Word := NativeKernel.reference I P R N x w epsilon j

def expected (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (R : NativeKernel.Reductions)
    (shape : Shape) (st : Strides) (epsilon : NativeKernel.Term)
    (xs : (layernormIO R shape st epsilon).Inputs) :
    (layernormIO R shape st epsilon).ObservedOutputs NativeKernel.Word :=
  fun o => Fin.cases (fun j => formula I P R shape.columns
    (KernelPublicIO.xRow (NativeKernel.algebra R) shape st epsilon xs (j.1,j.2.1))
    (KernelPublicIO.wRow (NativeKernel.algebra R) shape st epsilon xs (j.1,j.2.1)) epsilon j.2.2.1.val)
    (fun z => Fin.elim0 z) o

/-- Safe successful full-grid source execution, independent observed native
words, and complete frame. No floating transformation law is assumed. -/
theorem layernorm_fwd_triton_kernel_native (I : NativeKernel.Instructions)
    (P : NativeKernel.Profile) (R : NativeKernel.Reductions) (shape : Shape) (st : Strides)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val))
    (epsilon : NativeKernel.Term) :
    (layernormIO R shape st epsilon).Observes (observe I P R shape st epsilon)
      (expected I P R shape st epsilon) := by
  have projected := ProgramIO.Observes.of_implements
    (KernelPublicIO.implements (NativeKernel.algebra R) shape st epsilon injective
      (fun x w j => Graph.output (NativeKernel.algebra R) shape.columns x w epsilon j)
      (fun _ _ _ _ => rfl)) (observe I P R shape st epsilon)
  have same : (fun vals => (KernelPublicIO.io (NativeKernel.algebra R) shape st epsilon).mapOutputs
      (observe I P R shape st epsilon)
      (KernelPublicIO.target (NativeKernel.algebra R) shape st epsilon
        (fun x w j => Graph.output (NativeKernel.algebra R) shape.columns x w epsilon j) vals)) =
        expected I P R shape st epsilon := by
    funext vals o
    fin_cases o
    rfl
  rw [same] at projected
  exact projected

#specNonCircular formula avoiding [Source.kernel,Source.launch,Source.yValue,Source.referenceMem]
#axiomsClean layernorm_fwd_triton_kernel_native
#stmtSurfaceSubset layernorm_fwd_triton_kernel_native ⊆
  [NativeKernel.Instructions,NativeKernel.Profile,NativeKernel.Reductions,NativeKernel.Term,
   NativeKernel.Word,VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm,
   Shape,Shape.columns,Strides,Coordinate,Row,yOffset,layernormIO,observe,expected,ProgramIO.Observes]
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Equivalence
