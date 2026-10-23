import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.PublicFPResult
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.NativeInstructions

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000

/-- Pure native word observation. Symbolic inputs remain arbitrary expressions. -/
def nativeObserver (c : Call) (r : Regions) (ops : NativeInstructions.ScalarOps) :
    (o : Fin (outputs c r).length) → TileIndex (outputs c r)[o].shape →
      NativePhysical.Expr → BitVec 32 := fun _ _ e=>
  NativeInstructions.evaluate ops (NativeProfiles.plan c.q.dims) e (NativeProfiles.ownerOf c.q.dims e)

def nativeExpected (c : Call) (r : Regions) (referenceOps : NativeInstructions.ScalarOps)
    (xs : ChannelValues NativePhysical.Expr (inputs c r)) : ChannelValues (BitVec 32) (outputs c r) :=
  fun out => match out with
  | ⟨0,_⟩ => fun j =>
      let e := IOLogical.grid c r (NativePhysical.backend (NativeProfiles.plan c.q.dims))
        (FPHost.jobs c) (initialMemory c r xs) r.out (address c c.o j)
      NativeInstructions.evaluate referenceOps (NativeProfiles.plan c.q.dims) e
        (NativeProfiles.ownerOf c.q.dims e)

theorem native_complete (c : Call) (r : Regions) (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (laws : NativeInstructions.ScalarLaws actual referenceOps) :
    (io c r (NativePhysical.backend (NativeProfiles.plan c.q.dims))).ObservesOn
      (nativeObserver c r actual) (domain c r) (nativeExpected c r referenceOps) := by
  rw [laws.operations_eq actual referenceOps]
  have h := ProgramIO.ObservesOn.of_implementsOn
    (complete c r (NativePhysical.backend (NativeProfiles.plan c.q.dims)) startLabel lengthLabel)
    (nativeObserver c r referenceOps)
  have target : (fun xs=>(io c r (NativePhysical.backend (NativeProfiles.plan c.q.dims))).mapOutputs
      (nativeObserver c r referenceOps) (expected c r (NativePhysical.backend (NativeProfiles.plan c.q.dims)) xs))=
      nativeExpected c r referenceOps := by
    funext xs out j
    fin_cases out
    rfl
  rw [target] at h
  exact h

#specNonCircular nativeExpected avoiding [FPReference.grid,FPReference.runGrid,FP.Structural.exec,kernel]
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
