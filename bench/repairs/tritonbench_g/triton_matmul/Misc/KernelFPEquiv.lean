import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelNative

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelFPEquiv
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter KernelContract
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def expected (R : Native.Reference) := KernelNative.expected (Native.referenceInstructions R)

/-- Whole source, finite execution and physical bytes. The only numerical
premises compare fixed K16/K32 MMA instructions and the two output encodings.
The variable-shape dot law is derived, not required from callers. -/
specification kernel_native_if_local (rest : Algebra Word) (I : Native.Instructions) (R : Native.Reference)
    (numeric : Native.Correspondence I R) (read : Read) (a b c : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State Word) (output : Output a b c)
    (bound : TensorBinding.Inputs read id a b initial)
    (outputBound : OutputBytes.Bound read (storage a.metadata) c initial) :
    ∃ actual typed bytes,
      FiniteGrid.grid (Native.algebra rest I) (Native.dotAcc I) (request a b)
        (List.range (programs (request a b))) initial=some actual ∧
      runPrograms (Native.algebra rest I) (Native.dotAcc I) (request a b)
        (List.range (programs (request a b))) initial=some typed ∧
      actual.mem=typed.mem ∧
      ByteExecution.grid (Native.algebra rest I) (Native.dotAcc I) (request a b) c
        (List.range (programs (request a b))) (initial,read)=some (actual,bytes) ∧
      CheckedHost.Post a b c initial actual (expected R read a b la lb) ∧
      ByteHost.Post read bytes a b c output.tensor (expected R read a b la lb) := by
  obtain ⟨actual,typed,bytes,ha,ht,hm,hb,logical,physical⟩ :=
    KernelNative.correct rest I read a b c la lb h initial output bound outputBound
  have instruction := Native.instructions_eq I R numeric
  exact ⟨actual,typed,bytes,ha,ht,hm,hb,
    by simpa only [expected,instruction] using logical,
    by simpa only [expected,instruction] using physical⟩

#specNonCircular expected avoiding [Source.typed,FP.Structural.exec,runPrograms,FiniteGrid.grid,ByteExecution.grid]
#axiomsClean kernel_native_if_local
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelFPEquiv
