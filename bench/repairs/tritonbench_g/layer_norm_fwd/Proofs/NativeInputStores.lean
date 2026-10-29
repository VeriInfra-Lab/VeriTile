import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeInputValues
import bench.tritonbench_g.layer_norm_fwd.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

def keep {α : Type} (a : Ops α) (seed : Int) (p : α) (col : Nat) : Bool :=
  a.greater ((Expr.random seed col).eval a) ((Expr.input p).eval a)

theorem storeMask_read {α : Type} (f : Flags) (r : Regions) (index N block : Nat)
    (values : StreamRegisters α block) (s : State α) (hd : f.dropout=Bool.true) (hm : f.dropoutMask=Bool.true)
    (j : Fin block) (hj : j.val<N) :
    (storeMask f r index N block values s).mem r.dropoutMask (index*N+j.val)=Cell.mk .bool (values.keep j) := by
  simp only [storeMask,hd,hm,Bool.true_and,ite_true]
  exact vectorStore_read _ _ _ _ _ _ j hj

theorem input_residual_store {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) (hsep : Domain.ReadWriteSeparated f r)
    (ho : f.residualOut=Bool.true) (j : Fin block) (hj : j.val<N) :
    (inputStage a f r stride M N row p block s).memory.mem r.residualOut (row*stride.residualOut+j.val)=
      Cell.mk .real (a.cast32 ((NativeParameters.combined f (snapshot r stride s) M N p row j.val).eval a)) := by
  have hv := inputStage_value a f r stride M N row p block s hsep j
  dsimp only [inputStage]
  rw [if_pos ho,vectorStore_read _ _ _ _ _ _ j hj]
  exact congrArg (fun e : Expr α => Cell.mk .real (a.cast32 (e.eval a))) hv

theorem input_mask_store {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) (hrow : row<M)
    (hsep : r.dropoutMask≠r.residualOut) (hd : f.dropout=Bool.true) (hm : f.dropoutMask=Bool.true)
    (j : Fin block) (hj : j.val<N) :
    (inputStage a f r stride M N row p block s).memory.mem r.dropoutMask (row*N+j.val)=
      Cell.mk .bool (keep a ((snapshot r stride s).seeds row) p j.val) := by
  have hhalf : ¬Contract.inRow ((M+row)*N) N (row*N+j.val) :=
    fun h => WholeGrid.mask_halves_disjoint hrow (Contract.inRow_of_col hj) h
  dsimp only [inputStage]
  cases ho : f.residualOut <;> cases hx : f.x1
  all_goals try simp only [ho,hx,Bool.false_eq_true,ite_false,ite_true]
  all_goals try rw [vectorStore_frame _ r.residualOut _ _ _ _ r.dropoutMask (row*N+j.val) (by simp [hsep])]
  all_goals try rw [storeMask_preserves f r (M+row) N block _ _ r.dropoutMask (row*N+j.val) (by intros; simp [hhalf])]
  all_goals rw [storeMask_read _ _ _ _ _ _ _ hd hm j hj]
  all_goals rfl

theorem input_mask1_store {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) (hsep : Domain.ReadWriteSeparated f r)
    (hmres : r.dropoutMask≠r.residualOut) (hd : f.dropout=Bool.true) (hm : f.dropoutMask=Bool.true)
    (hx : f.x1=Bool.true) (j : Fin block) (hj : j.val<N) :
    (inputStage a f r stride M N row p block s).memory.mem r.dropoutMask ((M+row)*N+j.val)=
      Cell.mk .bool (keep a ((snapshot r stride s).seeds (M+row)) p j.val) := by
  have hseed : r.seeds≠r.dropoutMask := hsep _ (by simp [Domain.readRegion,hd]) _ (by simp [Domain.writeRegion,hd,hm])
  dsimp only [inputStage]
  rw [if_pos hx]
  cases ho : f.residualOut
  all_goals try simp only [ho,Bool.false_eq_true,ite_false,ite_true]
  all_goals try rw [vectorStore_frame _ r.residualOut _ _ _ _ r.dropoutMask ((M+row)*N+j.val) (by simp [hmres])]
  all_goals rw [storeMask_read _ _ _ _ _ _ _ hd hm j hj]
  all_goals dsimp only [stream,keep,snapshot]
  all_goals rw [storeMask_preserves f r row N block _ _ r.seeds (M+row) (by intros; simp [hseed])]

#axiomsClean input_residual_store
#axiomsClean input_mask_store
#axiomsClean input_mask1_store
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
