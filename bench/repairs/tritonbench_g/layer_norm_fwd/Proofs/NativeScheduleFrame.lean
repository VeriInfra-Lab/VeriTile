import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeSchedule

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow TypedSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem storeMask_preserves {α : Type} (f : Flags) (r : Regions) (index N block : Nat)
    (values : StreamRegisters α block) (s : State α) (region : RegionName) (offset : Nat)
    (h : f.dropout=Bool.true → f.dropoutMask=Bool.true →
      ¬(region=r.dropoutMask ∧ Contract.inRow (index*N) N offset)) :
    (storeMask f r index N block values s).mem region offset=s.mem region offset := by
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp only [storeMask,hd,hm,Bool.false_and,Bool.true_and,ite_false,ite_true]
  all_goals first | rfl | exact vectorStore_frame _ _ _ _ _ _ _ _ (h hd hm)

theorem storeMask_reads {α : Type} (f : Flags) (r : Regions) (index N block : Nat)
    (values : StreamRegisters α block) (s : State α) (hsep : Domain.ReadWriteSeparated f r) :
    ReadsEqual f r (storeMask f r index N block values s) s := by
  intro region hr offset
  apply storeMask_preserves
  intro hd hm h
  exact hsep region hr r.dropoutMask (by simp [Domain.writeRegion,hd,hm]) h.1

theorem inputStage_frame {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) :
    Frame f r stride M N row s (inputStage a f r stride M N row p block s).memory := by
  intro region offset hw
  have hres : f.residualOut=Bool.true → ¬(region=r.residualOut ∧ Contract.inRow (row*stride.residualOut) N offset) := by
    intro hf h; apply hw; simp [Contract.written,hf,h.1,h.2]
  have hm : f.dropout=Bool.true → f.dropoutMask=Bool.true →
      ¬(region=r.dropoutMask ∧ Contract.inRow (row*N) N offset) := by
    intro hd hm h; apply hw; simp [Contract.written,hd,hm,h.1,h.2]
  have hm1 : f.x1=Bool.true → f.dropout=Bool.true → f.dropoutMask=Bool.true →
      ¬(region=r.dropoutMask ∧ Contract.inRow ((M+row)*N) N offset) := by
    intro hx hd hm h; apply hw; simp [Contract.written,hx,hd,hm,h.1,h.2]
  dsimp only [inputStage]
  cases ho : f.residualOut <;> cases hx : f.x1
  all_goals try simp only [ho,hx,Bool.false_eq_true,ite_false,ite_true]
  all_goals try rw [vectorStore_frame _ r.residualOut _ _ _ _ region offset (hres ho)]
  all_goals try rw [storeMask_preserves f r (M+row) N block _ _ region offset (hm1 hx)]
  all_goals exact storeMask_preserves f r row N block _ s region offset hm

theorem normalizationStage_frame {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat) (eps : α) (hblock : 0<block)
    (input : InputRegisters α block) :
    Frame f r stride M N row input.memory (normalizationStage a red f r N row eps hblock input).memory := by
  intro region offset hw
  have hs : ¬(region=r.rstd ∧ offset=row) := by
    intro h; apply hw; simp [Contract.written,h.1,h.2]
  have hm : f.rms=Bool.false → ¬(region=r.mean ∧ offset=row) := by
    intro hf h; apply hw; simp [Contract.written,hf,h.1,h.2]
  dsimp only [normalizationStage]
  rw [scalarStore_frame _ r.rstd row _ region offset hs]
  cases hf : f.rms
  · try simp only [hf,Bool.false_eq_true,ite_false]
    exact scalarStore_frame _ r.mean row _ region offset (hm hf)
  · rfl

theorem affineStage_frame {α : Type} [Inhabited α] (a : Ops α) (plan plan1 : AffinePlan)
    (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat) (input : NormalRegisters α block) :
    Frame f r stride M N row input.memory (affineStage a plan plan1 f r stride N row input) := by
  intro region offset hw
  have hy : ¬(region=r.y ∧ Contract.inRow (row*stride.y) N offset) := by
    intro h; apply hw; simp [Contract.written,h.1,h.2]
  have hy1 : f.w1=Bool.true → ¬(region=r.y1 ∧ Contract.inRow (row*stride.y1) N offset) := by
    intro hf h; apply hw; simp [Contract.written,hf,h.1,h.2]
  dsimp only [affineStage]
  cases hf : f.w1
  · try simp only [hf,Bool.false_eq_true,ite_false]
    exact vectorStore_frame _ r.y _ _ _ _ region offset hy
  · try simp only [hf,ite_true]
    rw [vectorStore_frame _ r.y1 _ _ _ _ region offset (hy1 hf)]
    exact vectorStore_frame _ r.y _ _ _ _ region offset hy

theorem execute_frame {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (hblock : 0<block) (s : State α) :
    Frame f r stride M N row s (execute a red plan plan1 f r stride M N row eps p hblock s) :=
  (inputStage_frame a f r stride M N row p block s).trans
    ((normalizationStage_frame a red f r stride M N row eps hblock _).trans
      (affineStage_frame a plan plan1 f r stride M N row _))

theorem execute_inputs {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (hblock : 0<block) (s : State α) (hsep : Domain.ReadWriteSeparated f r) :
    ReadsEqual f r (execute a red plan plan1 f r stride M N row eps p hblock s) s :=
  (execute_frame a red plan plan1 f r stride M N row eps p hblock s).readsEqual hsep

#axiomsClean storeMask_reads
#axiomsClean inputStage_frame
#axiomsClean normalizationStage_frame
#axiomsClean affineStage_frame
#axiomsClean execute_frame
#axiomsClean execute_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
