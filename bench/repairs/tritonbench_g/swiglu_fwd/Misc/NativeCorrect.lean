import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.PublicIO
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Bench.Repairs.SwigluTriton.Native
noncomputable section

/-- Actual ordered native graph, retaining sigmoid constants and instruction
ports. These are machine primitives; no floating rewrite is assumed. -/
def fp32Formula (instructions : InstructionSemantics) (x y : Word) : Word :=
  let s := sigmoidGraph instructions x
  instructions.mulRN (instructions.mulRN s x) y

abbrev swigluNativeIO (instructions : InstructionSemantics) (c : Layout) (B row tile : Nat) : ProgramIO Word (Term Word) where
  inputs := [⟨.real,[B],c.X⟩,⟨.real,[B],c.Y⟩]
  outputs := [⟨.real,[B],c.OUT⟩]
  scratch := []
  buffers := [c.X,c.Y,c.OUT]
  read := fun i => Fin.cases (fun _ j => row*c.sx+(tile*B+j.1.val))
    (fun k => Fin.cases (fun _ j => row*c.sy+(tile*B+j.1.val)) (fun z => Fin.elim0 z) k) i
  inputMask := fun i => Fin.cases (fun _ j => tile*B+j.1.val<c.cols)
    (fun k => Fin.cases (fun _ j => tile*B+j.1.val<c.cols) (fun z => Fin.elim0 z) k) i
  write := fun o => Fin.cases (fun _ j => row*c.so+(tile*B+j.1.val)) (fun z => Fin.elim0 z) o
  outputMask := fun o => Fin.cases (fun _ j => tile*B+j.1.val<c.cols) (fun z => Fin.elim0 z) o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [{
    kernel := kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B
    backend := graphAlgebra
    pids := fun ax => if ax=0 then row else if ax=1 then tile else 0 }]
  encode := Term.input
  observe := fun _ _ => interpret (compiledPorts instructions)

theorem swiglu_kernel_native (instructions : InstructionSemantics)
    (c : Layout) (B row tile : Nat) :
    (swigluNativeIO instructions c B row tile).Implements
      (KernelPublic.target graphAlgebra Term.input (interpret (compiledPorts instructions))
        c B row tile (fp32Formula instructions)) := by
  apply KernelPublic.implements
  intro x y
  rfl

#specNonCircular fp32Formula avoiding [kernel,interpret,scalar]
#axiomsClean swiglu_kernel_native
#stmtSurfaceSubset swiglu_kernel_native ⊆
  [InstructionSemantics, Layout, Word, Term, swigluNativeIO, ProgramIO.Implements,
   KernelPublic.target, graphAlgebra, Term.input, interpret, compiledPorts, fp32Formula,
   VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.instInhabitedTerm]

end
end VeriTile.Bench.Repairs.SwigluFwd
