import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.ByteGrid

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.RawNative
open VeriTile.Triton FP.Structural Execution
noncomputable section

/-- Complete native execution over raw bytes, original scalar/K16 instruction
parameters, independent ordered half-word outputs and all exterior bytes. -/
theorem kernel_equiv (a : Words.Instructions) (p : Parameters) (initial : Words.Memory) :
    ∃ final,ByteGrid.run a p initial=some final ∧
      ByteGrid.Post a p initial (Words.commit initial final) :=
  ByteGrid.complete a p initial

theorem empty_native (a : Words.Instructions) (p : Parameters) (h : p.rows=0)
    (initial : Words.Memory) :
    ByteGrid.run a p initial=some (Words.entry p initial) ∧
      Words.commit initial (Words.entry p initial)=initial :=
  ⟨Grid.empty_grid Bool.true _ _ p _ h,Words.commit_entry p initial⟩

#axiomsClean empty_native
#axiomsClean kernel_equiv
#specNonCircular ByteGrid.Post avoiding [ByteGrid.run,Grid.grid,Grid.one,nativeExecution,source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.RawNative
