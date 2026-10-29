import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawBackendTile
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program Backend
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem limit_eq (a b : Nat) : (if a < b then a else b) = min a b := by
  split_ifs <;> omega

theorem sourceRun_mem {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) :
    (sourceRun c p r hasBias useSoftplus lower upper l s).val.mem =
      tileMemory c p r hasBias useSoftplus lower upper l s := by
  cases hasBias <;> cases useSoftplus
  all_goals simp only [sourceRun,store,scatter_mem]
  all_goals simp [tileMemory,scatter,preparedTile,prepare,dtRaw,biasRaw,rateRaw,activeInput,activeOutput,
    dtAddress,outAddress,head,Whole.Host.layout,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,WholeFP.tileState,State.setReg,decide_eq_true_eq,limit_eq]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
