import bench.repairs.tritonbench_g.chunk_linear_attn.IOLayout

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOLayoutFacts
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} (c : KernelInterface.Config) (vals : ChannelValues α (IOLayout.inputs c.dims))

theorem native_bounds (bounds : RegionName → Nat)
    (backed : (IOLayout.nativeMemory α c).InBounds bounds vals) :
    (IOLayout.realMemory α c).InBounds bounds vals := by
  refine ⟨backed.1,backed.2.1,?_,backed.2.2.2⟩
  intro o j enabled
  fin_cases o
  · exact backed.2.2.1 ⟨0,by change 0 < 9; decide⟩ j enabled
  · exact backed.2.2.1 ⟨1,by change 1 < 9; decide⟩ j enabled
  · exact backed.2.2.1 ⟨2,by change 2 < 9; decide⟩ j enabled
  · exact backed.2.2.1 ⟨3,by change 3 < 9; decide⟩ j enabled
  · exact backed.2.2.1 ⟨4,by change 4 < 9; decide⟩ j enabled
  · exact backed.2.2.1 ⟨5,by change 5 < 9; decide⟩ j enabled

theorem native_outside (r : RegionName) (a : Nat)
    (outside : (IOLayout.nativeMemory α c).Outside vals r a) :
    (IOLayout.realMemory α c).Outside vals r a := by
  refine ⟨?_,outside.2⟩
  intro o j enabled
  fin_cases o
  · exact outside.1 ⟨0,by change 0 < 9; decide⟩ j enabled
  · exact outside.1 ⟨1,by change 1 < 9; decide⟩ j enabled
  · exact outside.1 ⟨2,by change 2 < 9; decide⟩ j enabled
  · exact outside.1 ⟨3,by change 3 < 9; decide⟩ j enabled
  · exact outside.1 ⟨4,by change 4 < 9; decide⟩ j enabled
  · exact outside.1 ⟨5,by change 5 < 9; decide⟩ j enabled

#axiomsClean native_bounds
#axiomsClean native_outside
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOLayoutFacts
