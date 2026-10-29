import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPReferenceGrid
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.NativeInstructions

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

def kernelWord (c : Call) (ops : NativeInstructions.ScalarOps)
    (s : State NativePhysical.Expr) (r : RegionName) (off : Nat) : BitVec 32 :=
  let e:=(s.mem r off).read .real
  NativeInstructions.evaluate ops (NativeProfiles.plan c.q.dims) e (NativeProfiles.ownerOf c.q.dims e)

/-- Complete source grid, independent online recurrence, selected SM80 native
word observations and exterior frame. The integer read correspondence concerns
raw length/start memory cells; no Tensor admission or binding is supplied. -/
theorem fp_equiv (c : Call) (r : Regions) (initial : State NativePhysical.Expr)
    (integerReads : FPHostSource.Metadata c r initial)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    let model:=NativePhysical.backend (NativeProfiles.plan c.q.dims)
    let order:=FPHost.jobs c
    ∃ final,FPReference.runGrid c r model order initial=some final ∧
      final.mem=(FPReference.grid c r model order initial).mem ∧
      (∀ region offset,kernelWord c actual final region offset=
        kernelWord c referenceOps (FPReference.grid c r model order initial) region offset) ∧
      (∀ region offset,¬FPReference.Writes c r order region offset →
        final.mem region offset=initial.mem region offset) := by
  dsimp only
  obtain ⟨final,run,memory⟩:=FPReference.whole c r
    (NativePhysical.backend (NativeProfiles.plan c.q.dims)) (FPHost.jobs c) initial
  refine ⟨final,run,memory,?_,?_⟩
  · intro region offset
    rw [kernelWord,kernelWord,memory,localLaws.operations_eq actual referenceOps]
  · intro region offset outside
    rw [memory]
    exact (FPReference.grid_frame c r _ (FPHost.jobs c) initial integerReads startLabel lengthLabel).2 region offset outside

#axiomsClean fp_equiv
#specNonCircular FPReference.grid avoiding [FPReference.runGrid,FP.Structural.exec,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.KernelNativeResult
