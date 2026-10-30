import bench.tritonbench_g.streamk_matmul.Proofs.AreaEquivalence

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.AreaEquivalence
open Whole001 Source Reader VeriTile Triton
noncomputable section

/-- The same actual complete tape supplies both implementations' successful
runs, source accesses, observations and full exterior frames. -/
theorem equivalent (reference : Native.Reference) (p : Parameters) :
    ProgramView.Equivalent
      (⟨kernels⟩ : ProgramView.Implementation reference p) ⟨areaKernels⟩ := by
  intro input before binding
  obtain ⟨program,decoded,events,final,ran,safe,output,frame⟩ :=
    EventIO.native_implements reference p input trivial before binding
  have candidateDecoded :
      (ProgramView.sourceTape areaKernels p).map Concurrent.Program.tape = some program := by
    rw [← source_tape]
    exact decoded
  exact ⟨program,program,events,final,EventIO.nativeFormula reference p input,
    decoded,candidateDecoded,ran,ran,safe,safe,output,output,frame,frame⟩

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.AreaEquivalence
