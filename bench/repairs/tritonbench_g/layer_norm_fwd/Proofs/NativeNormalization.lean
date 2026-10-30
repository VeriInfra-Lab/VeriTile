import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeInputValues

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem normalization_mean {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (M N row : Nat) (eps p : α) (hblock : 0<block)
    (input : InputRegisters α block) (x : NativeRow.Inputs α)
    (hx : ∀ j, input.value j=NativeParameters.combined f x M N p row j.val) (owner : Nat) :
    (normalizationStage a red f r N row eps hblock input).mean owner=
      NativeParameters.mean red f x M N p row owner := by
  have hv := funext hx
  simp only [normalizationStage,NativeParameters.mean,hv]

theorem normalization_centered {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (M N row : Nat) (eps p : α) (hblock : 0<block)
    (input : InputRegisters α block) (x : NativeRow.Inputs α)
    (hx : ∀ j, input.value j=NativeParameters.combined f x M N p row j.val) (j : Fin block) :
    (normalizationStage a red f r N row eps hblock input).centered j=
      NativeParameters.centered red f x M N p row j.val := by
  have hv := funext hx
  simp only [normalizationStage,NativeParameters.centered,NativeParameters.mean,hv]

/-- Each variance leaf uses its own mean owner. Equality here is equality of
pending graph constructors, before any opaque floating primitive is evaluated. -/
theorem normalization_rstd {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (M N row : Nat) (eps p : α) (hblock : 0<block)
    (input : InputRegisters α block) (x : NativeRow.Inputs α)
    (hx : ∀ j, input.value j=NativeParameters.combined f x M N p row j.val) (owner : Nat) :
    (normalizationStage a red f r N row eps hblock input).rstd owner=
      NativeParameters.rstd red f x M N eps p row owner := by
  have hv := funext hx
  simp only [normalizationStage,NativeParameters.rstd,NativeParameters.variance,NativeParameters.centered,NativeParameters.mean,hv]

theorem normalization_rstd_store {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (M N row : Nat) (eps p : α) (hblock : 0<block)
    (input : InputRegisters α block) (x : NativeRow.Inputs α)
    (hx : ∀ j, input.value j=NativeParameters.combined f x M N p row j.val) :
    (normalizationStage a red f r N row eps hblock input).memory.mem r.rstd row=
      Cell.mk .real (a.cast32 ((NativeParameters.rstd red f x M N eps p row 0).eval a)) := by
  have hg := normalization_rstd a red f r M N row eps p hblock input x hx 0
  change (scalarStore _ r.rstd row (.mk .real
    (a.cast32 (((normalizationStage a red f r N row eps hblock input).rstd 0).eval a)))).mem r.rstd row=_
  rw [scalarStore_read,hg]

theorem normalization_mean_store {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (M N row : Nat) (eps p : α) (hblock : 0<block)
    (input : InputRegisters α block) (x : NativeRow.Inputs α)
    (hx : ∀ j, input.value j=NativeParameters.combined f x M N p row j.val)
    (hrms : f.rms=Bool.false) (hsep : r.mean≠r.rstd) :
    (normalizationStage a red f r N row eps hblock input).memory.mem r.mean row=
      Cell.mk .real (a.cast32 ((NativeParameters.mean red f x M N p row 0).eval a)) := by
  have hg := normalization_mean a red f r M N row eps p hblock input x hx 0
  dsimp only [normalizationStage]
  rw [scalarStore_frame _ r.rstd row _ r.mean row (by simp [hsep])]
  simp only [hrms,Bool.false_eq_true,ite_false]
  rw [scalarStore_read]
  exact congrArg (fun e : Expr α => Cell.mk .real (a.cast32 (e.eval a))) hg

theorem normalization_preserves {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (N row : Nat) (eps : α) (hblock : 0<block)
    (input : InputRegisters α block) (region : RegionName) (offset : Nat)
    (hstd : region≠r.rstd) (hmean : region≠r.mean) :
    (normalizationStage a red f r N row eps hblock input).memory.mem region offset=input.memory.mem region offset := by
  dsimp only [normalizationStage]
  rw [scalarStore_frame _ r.rstd row _ region offset (by simp [hstd])]
  cases hf : f.rms
  · rw [if_neg Bool.false_ne_true]
    exact scalarStore_frame _ r.mean row _ region offset (by simp [hmean])
  · rfl

#axiomsClean normalization_mean
#axiomsClean normalization_centered
#axiomsClean normalization_rstd
#axiomsClean normalization_rstd_store
#axiomsClean normalization_mean_store
#axiomsClean normalization_preserves
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
