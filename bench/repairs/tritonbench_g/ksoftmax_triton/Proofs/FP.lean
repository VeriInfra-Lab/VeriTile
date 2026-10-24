import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Source
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedTotal

namespace VeriTile.Bench.Repairs.Ksoftmax.Floating
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def backward_run {α : Type} [Inhabited α] (M : Algebra α)
    (Dst G O : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal : Bool) (s : State α) :
    {t : State α // run (engine M) (backward Dst G O bm bn gm gn om on K D log causal).surfaceBody s=some t} := by
  cases log <;> cases causal
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [backward,ComputeKernel.surfaceBody,run,step,stepAlg,
      FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,numericLt,numericLe,natLt,natLe,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store]
    try rfl

/-- Reuse the existing valid total-execution theorem for the unchanged forward
arithmetic and control projection. Packing and native addressing are separate. -/
def forward_run {α : Type} [Inhabited α] (M : Algebra α)
    (Y X Mask : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0<D)
    (log : Bool) (mode : Forward.MaskMode) (causal : Bool) (s : State α) :
    {t : State α // run (engine M)
      (forward Y X Mask ym yn xm xn sm K D log mode.python causal Bool.false).surfaceBody s=some t} :=
  Forward.TypedTotal.fp32_source_run M Y X Mask ym yn xm xn sm K D hD log mode causal Bool.false s

theorem backward_frame {α : Type} [Inhabited α] (M : Algebra α)
    (Dst G O : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal : Bool) (s : State α) (r : RegionName) (offset : Nat)
    (outside : r≠Dst ∨ ∀ i : Fin D, i.val<K → offset≠s.pids 0*bm+s.pids 1*bn+i.val) :
    (backward_run M Dst G O bm bn gm gn om on K D log causal s).val.mem r offset=s.mem r offset := by
  cases log <;> cases causal
  all_goals
    dsimp only [backward_run]
    refine Eq.trans (State.masked_scatter_frame _ _ _ _ _ r offset ?_ _) rfl
    rcases outside with h|h
    · exact Or.inl h
    · exact Or.inr (fun i _ hi => h i.1 hi)

theorem forward_frame {α : Type} [Inhabited α] (M : Algebra α)
    (Y X Mask : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0<D)
    (log : Bool) (mode : Forward.MaskMode) (causal : Bool) (s : State α)
    (r : RegionName) (offset : Nat)
    (outside : r≠Y ∨ ∀ i : Fin D, i.val<K → offset≠s.pids 0*ym+s.pids 1*yn+i.val) :
    (forward_run M Y X Mask ym yn xm xn sm K D hD log mode causal s).val.mem r offset=s.mem r offset := by
  cases mode <;> cases log <;> cases causal
  all_goals
    dsimp only [forward_run,Forward.TypedTotal.fp32_source_run]
    refine Eq.trans (State.masked_scatter_frame _ _ _ _ _ r offset ?_ _) rfl
    rcases outside with h|h
    · exact Or.inl h
    · exact Or.inr (fun i _ hi => h i.1 hi)

/-- The repaired causal future output depends only on the FP32 zero literal
and destination cast, even if unread future input words contain NaN/infinity. -/
theorem backward_future {α : Type} [Inhabited α] (M : Algebra α)
    (Dst G O : RegionName) (bm bn gm gn om on K D : Nat) (log : Bool)
    (s : State α) (i : Fin D) (written : i.val<K) (future : s.pids 1 < i.val) :
    (backward_run M Dst G O bm bn gm gn om on K D log Bool.true s).val.mem Dst
      (s.pids 0*bm+s.pids 1*bn+i.val)=
      Cell.mk .real (M.cast (some .fp32) .real .real (M.literal (some .fp32) .real 0)) := by
  cases log
  all_goals
    dsimp only [backward_run]
    rw [State.masked_scatter_readback (shape := [D]) Dst
      (fun j => s.pids 0*bm+s.pids 1*bn+j.1.val) _ (fun j => j.1.val<K) _
      (i,PUnit.unit) written
      (by intro a b h;rcases a with ⟨a,⟨⟩⟩;rcases b with ⟨b,⟨⟩⟩;congr 1;apply Fin.ext;simpa using h)]
    have hz : (0.0 : ℝ)=0 := by norm_num
    simp [Nat.not_le.mpr future,hz]

#axiomsClean backward_future
#axiomsClean backward_run
#axiomsClean forward_run
#axiomsClean backward_frame
#axiomsClean forward_frame
end
end VeriTile.Bench.Repairs.Ksoftmax.Floating
