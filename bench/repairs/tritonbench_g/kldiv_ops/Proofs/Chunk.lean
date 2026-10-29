import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Tile

namespace VeriTile.Bench.Repairs.KldivOps.Tile
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem masked_regs {α ι : Type} (items : List ι) (active : ι→Prop)
    [DecidablePred active] (r : RegionName) (address : ι→Nat) (value : ι→Cell α) (s : State α) :
    (items.foldl (fun t i => if active i then t.write r (address i) (value i) else t) s).regs=s.regs := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> rfl

theorem chunk_registers {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    Registers p red row eps (nextCarry M p flag red eps carry row start s)
      (chunkRun M p flag red row start eps carry s h).val := by
  cases flag <;> by_cases hr : red=0
  all_goals constructor
  all_goals
    simp [chunkRun,nextCarry,chunkSum,lane,scalar,zero,hr,masked_regs,
      State.setReg,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,Int.toNat_add,active_int,ofFloat,toFloat,engine,
      Algebra.withDefaultPrecision,resolvePrecision,h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,h.cols]
  all_goals try simp only [←neg_add,←Int.natCast_mul,←Int.natCast_add,Int.zero_sub,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]
  all_goals norm_num
  all_goals funext i; cases i; rfl

theorem chunk_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps : α) (s : State α) (r : RegionName) (i : Nat)
    (outside : r≠p.O ∨ ∀ j : TileIndex [p.block], start+j.1.val<p.cols → i≠row*p.cols+(start+j.1.val)) :
    chunkMemory M p flag red row start eps s r i=s.mem r i := by
  unfold chunkMemory
  split
  · rw [←VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
    apply State.masked_scatter_frame
    exact outside.imp id (fun h j _ hj => h j hj)
  · rfl

end
end VeriTile.Bench.Repairs.KldivOps.Tile
