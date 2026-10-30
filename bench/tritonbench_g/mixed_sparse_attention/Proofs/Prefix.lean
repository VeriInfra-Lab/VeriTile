import bench.tritonbench_g.mixed_sparse_attention.Proofs.PrefixCore
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem prefix_ready {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α) :
    Ready M c s (prefixExec M D c s).val (initialAccumulator M c) := by
  rcases c with ⟨storage,q,k,v,out,seq,blocks,offsets,counts,columns,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svn,svk,soz,soh,som,sok,Z,H,T,NR,NS,NV,BM,BN,BD,scale⟩
  cases storage <;> dsimp only [prefixExec] <;> constructor <;>
    simp [TypedSource.Storage.tile,TypedSource.Storage.float,State.setReg,sequenceLength,rowStart,
      metadataIndex,blockCount,columnCount,queryBase,keyValueBase,queryValues,keyPointer,valuePointer,
      outputPointer,initialAccumulator,backend,bop,ofFloat,toFloat]
  all_goals first | rfl | (funext i; simp [bop,keyPointer,valuePointer,outputPointer,keyValueBase,queryBase,rowStart,queryValues,ofFloat,toFloat,Region.cast])

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α) :
    ∃ final, run (backend M D) (prelude c) s = some final ∧ Ready M c s final (initialAccumulator M c) :=
  ⟨(prefixExec M D c s).val,(prefixExec M D c s).property,prefix_ready M D c s⟩

#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
