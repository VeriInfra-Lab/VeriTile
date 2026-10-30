import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelValues
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton FP.Structural
open Math
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem limit_eq (a b : Nat) : (if a < b then a else b) = min a b := by
  split_ifs <;> omega

/-- The memory produced by the complete shared execution, after proving the
source's uint64 dimension and lane casts exact on each admitted program. -/
theorem sourceRun_mem {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (sizes : Sizes c) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (hl : c.valid l) (s : State α) :
    (sourceRun c p hasBias useSoftplus lower upper l s).val.mem =
      tileMemory c p hasBias useSoftplus lower upper l s := by
  obtain ⟨hn,hh,hc,hb,hk,hhead⟩ := scalar_mods c sizes l hl
  simp only [Nat.reducePow] at hn hh hc hb hk hhead
  have hrow (i : Fin c.BH) : i.val % 18446744073709551616 = i.val := row_mod c i
  have hcol (j : Fin c.BC) : j.val % 18446744073709551616 = j.val := column_mod c sizes j
  have hone : (1.0 : ℝ) = 1 := by norm_num
  cases hasBias <;> cases useSoftplus
  all_goals simp only [sourceRun,store,scatter_mem]
  all_goals simp [tileMemory,scatter,preparedTile,prepare,dtRaw,biasRaw,rateRaw,activeInput,activeOutput,
    dtAddress,outAddress,head,Geometry.layout,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,tileState,bindBounds,State.setReg,decide_eq_true_eq,limit_eq,
    integralMod,IntegerExpr.same_left,IntegerExpr.same_right,
    hn,hh,hc,hb,hk,hhead,hrow,hcol,hone,Nat.add_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
