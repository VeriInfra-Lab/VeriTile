import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.NativeInputConnection
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Proofs.PublicIO

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry records the remaining comparison blocker. -/

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Same complete source and strided matrix channels as Correct, with raw
FP32 words and the original selected fixed-k8 TF32 instruction family. -/
def nativeIO (ops : Native.Reference) (choice : Fin 8)
    (rest : Algebra Native.Word) (p : Core.Parameters) : ProgramIO Native.Word Native.Word :=
  IO.matmul rest
    (Native.primitives (Native.referenceInstructions ops) (Native.family choice)) p

/-- Independent padded instruction sequence. All FP32 words are admitted;
masked lanes retain zero and every block retains its incoming accumulator. -/
def nativeTarget (ops : Native.Reference) (choice : Fin 8) (K B : Nat)
    (a b : Nat → Nat → Native.Word) (r c : Nat) : Native.Word :=
  Reference.accumulation
    (Native.primitives (Native.referenceInstructions ops) (Native.family choice))
    K B a b r c ((K+B-1)/B)

/-- Complete native observation, preserving the prior source-to-reference
scope. The two fixed-k8 primitives are computational model parameters, not
transformation axioms or a claim that native arithmetic equals Real. -/
theorem kernel_native (ops : Native.Reference) (choice : Fin 8)
    (rest : Algebra Native.Word) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (input : p.input≠p.output) (weight : p.weight≠p.output) :
    (nativeIO ops choice rest p).Implements
      (IO.formula p (nativeTarget ops choice p.innerSize p.blockInner)) := by
  apply PublicIO.implements rest _ p geometry inner input weight
  intro initial a b ha hb r c hr hc
  exact KernelInputs.logical_reference _ p initial a b ha hb r c hr hc

#axiomsClean kernel_native
#specNonCircular nativeTarget avoiding [Core.grid,Core.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
