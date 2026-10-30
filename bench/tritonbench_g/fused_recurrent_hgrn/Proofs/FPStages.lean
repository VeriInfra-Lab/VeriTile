import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.fused_recurrent_hgrn.FusedRecurrentHgrn

/-! Typed fp32 contracts for the four already selected HGRN stages.
The carry is supplied through the same materialized stage input as the
existing public stage contracts. Each scalar arithmetic operation is selected
by its compute precision in the opaque backend. This file never uses real
ring operations to interpret the floating values.

The selected source profile has fp32 storage and an fp32 carry, with separate
multiply/add operations. A native specialization must establish the exact
load/cast/operation/store correspondence, including fusion and FTZ settings.
The stage cut and host/autograd boundaries are unchanged.
-/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def forward (BHPrev X G O : RegionName) (i_t T D BD : Nat) : ComputeKernel :=
  ComputeKernel.mk [BHPrev, X, G] [O] (triton {
  tl.region BHPrev = tl.float32, X = tl.float32, G = tl.float32, O = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  prev = tl.load(BHPrev + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  b_x = tl.load(X + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_g = tl.load(G + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_h = b_g * prev + b_x
  tl.store(O + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_h).to(O.dtype.element_ty), mask=mask)
}).surfaceBody

theorem forward_success {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev X G O : RegionName) (i_t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (forward BHPrev X G O i_t T D BD) s = some final := by
  simp [forward, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast]

def channel {α : Type} (s : State α) (BD : Nat) (i : TileIndex [BD]) : Nat :=
  s.pids 0 * BD + i.1.val

def rowOffset {α : Type} (s : State α) (t T D BD : Nat) (i : TileIndex [BD]) : Nat :=
  (s.pids 1 * T + t) * D + channel s BD i

def stateOffset {α : Type} (s : State α) (D BD : Nat) (i : TileIndex [BD]) : Nat :=
  s.pids 1 * D + channel s BD i

/-- Independent scalar recurrence: exact callback order, with no associativity,
distributivity, final-output premise or real-arithmetic interpretation. -/
def forwardValue {α : Type} (M : Algebra α) (carry gate input : α) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul
        (M.cast (some .fp32) .real .real gate) (M.fp32Load carry))
      (M.cast (some .fp32) .real .real input))

def maskedInput {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (offset : Nat) (active : Prop) [Decidable active] : α :=
  if active then (s.mem R offset).read .real else M.literal (some .fp32) .real 0

def forwardValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (BHPrev X G : RegionName) (t T D BD : Nat) : TileIndex [BD] → α :=
  fun i => forwardValue M
    (maskedInput M s BHPrev (stateOffset s D BD i) (channel s BD i < D))
    (maskedInput M s G (rowOffset s t T D BD i) (channel s BD i < D))
    (maskedInput M s X (rowOffset s t T D BD i) (channel s BD i < D))

def scatter {α : Type} (s : State α) (R : RegionName) (BD : Nat)
    (offset : TileIndex [BD] → Nat) (active : TileIndex [BD] → Prop)
    [DecidablePred active] (value : TileIndex [BD] → α) : State α :=
  (TileShape.allIndices [BD]).foldl
    (fun u i => if active i then u.write R (offset i) (.mk .real (value i)) else u) s

theorem scatter_memory_congr {α ι : Type} (l : List ι) (s u : State α)
    (R : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → α) (h : s.mem = u.mem) :
    (l.foldl (fun v i => if active i then v.write R (offset i) (.mk .real (value i)) else v) s).mem =
    (l.foldl (fun v i => if active i then v.write R (offset i) (.mk .real (value i)) else v) u).mem := by
  induction l generalizing s u with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      apply ih
      simp only [State.write, h]
    · simp only [hi, ↓reduceIte]
      exact ih s u h

theorem forward_memory {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (forward BHPrev X G O t T D BD) s = some final ∧
      final.mem = (scatter s O BD (rowOffset s t T D BD)
        (fun i => channel s BD i < D) (forwardValues M s BHPrev X G t T D BD)).mem := by
  simp only [forward, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, bind, pure, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast,
    State.setReg_same, State.setReg_ne_name, State.setReg_mem, State.setReg_pids,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  simp [State.setReg, forwardValues, forwardValue, maskedInput, scatter, bop,
    rowOffset, stateOffset, channel, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem rowOffset_injective {α : Type} (s : State α) (t T D BD : Nat) :
    Function.Injective (rowOffset s t T D BD) := by
  intro i j h
  have he : i.1 = j.1 := by
    apply Fin.ext
    dsimp [rowOffset, channel] at h
    omega
  exact Prod.ext he (Subsingleton.elim _ _)

/-- Successful typed output, exact per-operation scalar value, and every
unwritten tagged cell unchanged. The supplied input is the initial state. -/
def Post {α : Type} (s final : State α) (O : RegionName) (t T D BD : Nat)
    (values : TileIndex [BD] → α) : Prop :=
  (∀ i, channel s BD i < D → final.mem O (rowOffset s t T D BD i) = .mk .real (values i)) ∧
  (∀ R off, (R ≠ O ∨ ∀ i, channel s BD i < D → off ≠ rowOffset s t T D BD i) →
    final.mem R off = s.mem R off)

theorem post_of_memory {α : Type} (s final : State α) (O : RegionName) (t T D BD : Nat)
    (values : TileIndex [BD] → α)
    (hm : final.mem = (scatter s O BD (rowOffset s t T D BD)
      (fun i => channel s BD i < D) values).mem) : Post s final O t T D BD values := by
  constructor
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback O (rowOffset s t T D BD)
      (fun j => .mk .real (values j)) (fun j => channel s BD j < D) s i hi
      (rowOffset_injective s t T D BD)
  · intro R off h
    rw [hm]
    apply State.masked_scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => h i hi

specification forward_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (forwardValues M s BHPrev X G t T D BD) := by
  obtain ⟨final, he, hm⟩ := forward_memory M BHPrev X G O t T D BD s
  exact ⟨final, he, post_of_memory s final O t T D BD _ hm⟩

/-- Projection equality is only the old/new stage-shape bridge; the FP result
above is proved directly in Structural, never by this real projection. -/


def dx (DHPrev DO DX : RegionName) (t T D BD : Nat) : ComputeKernel :=
  ComputeKernel.mk [DHPrev, DO] [DX] (triton {
  tl.region DHPrev = tl.float32, DO = tl.float32, DX = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do
  tl.store(DX + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    (b_dh).to(DX.dtype.element_ty), mask=mask)
}).surfaceBody

def dxTwo (DHPrev DO G DX : RegionName) (t T D BD : Nat) : ComputeKernel :=
  ComputeKernel.mk [DHPrev, DO, G] [DX] (triton {
  tl.region DHPrev = tl.float32, DO = tl.float32, G = tl.float32, DX = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do_next = tl.load(DO + (i_bh * $(T) + $(t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_g_next = tl.load(G + (i_bh * $(T) + $(t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do_next
  b_dh = b_dh * b_g_next
  b_do = tl.load(DO + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = b_dh + b_do
  tl.store(DX + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    (b_dh).to(DX.dtype.element_ty), mask=mask)
}).surfaceBody

def dg (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) : ComputeKernel :=
  ComputeKernel.mk [DHPrev, DO, O, H0] [DG] (triton {
  tl.region DHPrev = tl.float32, DO = tl.float32, O = tl.float32,
    H0 = tl.float32, DG = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  if $(t) > $(0) {
    b_o = tl.load(O + (i_bh * $(T) + $(t - 1)) * $(D) + offs_d,
      mask=mask, other=0.0).to(tl.float32)
  } else {
    if U {
      b_o = tl.load(H0 + i_bh * $(D) + offs_d, mask=mask, other=0.0).to(tl.float32)
    } else {
      b_o = tl.zeros([$(BD)], dtype=tl.float32)
    }
  }
  b_dh = dh_prev + b_do
  b_dg = b_dh * b_o
  tl.store(DG + (i_bh * $(T) + $(t)) * $(D) + offs_d,
    (b_dg).to(DG.dtype.element_ty), mask=mask)
}).surfaceBody

def dxValue {α : Type} (M : Algebra α) (carry input : α) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add (M.fp32Load carry)
      (M.cast (some .fp32) .real .real input))

def dxTwoValue {α : Type} (M : Algebra α) (carry nextDo nextGate input : α) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .add (M.fp32Load carry)
          (M.cast (some .fp32) .real .real nextDo))
        (M.cast (some .fp32) .real .real nextGate))
      (M.cast (some .fp32) .real .real input))

def dxValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev DO : RegionName) (t T D BD : Nat) : TileIndex [BD] → α :=
  fun i => dxValue M
    (maskedInput M s DHPrev (rowOffset s t T D BD i) (channel s BD i < D))
    (maskedInput M s DO (rowOffset s t T D BD i) (channel s BD i < D))

def dxTwoValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev DO G : RegionName) (t T D BD : Nat) : TileIndex [BD] → α :=
  fun i => dxTwoValue M
    (maskedInput M s DHPrev (rowOffset s (t+1) T D BD i) (channel s BD i < D))
    (maskedInput M s DO (rowOffset s (t+1) T D BD i) (channel s BD i < D))
    (maskedInput M s G (rowOffset s (t+1) T D BD i) (channel s BD i < D))
    (maskedInput M s DO (rowOffset s t T D BD i) (channel s BD i < D))

def dgValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev DO O H0 : RegionName) (U : Bool) (t T D BD : Nat) : TileIndex [BD] → α :=
  fun i =>
    let prev := if 0 < t then
        M.cast (some .fp32) .real .real
          (maskedInput M s O (rowOffset s (t-1) T D BD i) (channel s BD i < D))
      else if U then M.cast (some .fp32) .real .real
        (maskedInput M s H0 (stateOffset s D BD i) (channel s BD i < D))
      else M.literal (some .fp32) .real 0
    M.cast (some .fp32) .real .real
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .add
          (M.fp32Load (maskedInput M s DHPrev (rowOffset s t T D BD i) (channel s BD i < D)))
          (M.cast (some .fp32) .real .real
            (maskedInput M s DO (rowOffset s t T D BD i) (channel s BD i < D)))) prev)

theorem dx_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dx DHPrev DO DX t T D BD) s = some final ∧
      final.mem = (scatter s DX BD (rowOffset s t T D BD)
        (fun i => channel s BD i < D) (dxValues M s DHPrev DO t T D BD)).mem := by
  simp [dx, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dxValues, dxValue, maskedInput,
    scatter, rowOffset, channel, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem dxTwo_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      final.mem = (scatter s DX BD (rowOffset s t T D BD)
        (fun i => channel s BD i < D) (dxTwoValues M s DHPrev DO G t T D BD)).mem := by
  simp [dxTwo, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dxTwoValues, dxTwoValue, maskedInput,
    scatter, rowOffset, channel, show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem dg_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      final.mem = (scatter s DG BD (rowOffset s t T D BD)
        (fun i => channel s BD i < D) (dgValues M s DHPrev DO O H0 U t T D BD)).mem := by
  cases U <;> by_cases ht : 0 < t
  all_goals simp [dg, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dgValues, maskedInput,
    scatter, rowOffset, stateOffset, channel, ht, show (0.0 : ℝ) = 0 by norm_num]
  all_goals apply scatter_memory_congr
  all_goals rfl

specification dx_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxValues M s DHPrev DO t T D BD) := by
  obtain ⟨final, he, hm⟩ := dx_memory M DHPrev DO DX t T D BD s
  exact ⟨final, he, post_of_memory s final DX t T D BD _ hm⟩

specification dx_two_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxTwoValues M s DHPrev DO G t T D BD) := by
  obtain ⟨final, he, hm⟩ := dxTwo_memory M DHPrev DO G DX t T D BD s
  exact ⟨final, he, post_of_memory s final DX t T D BD _ hm⟩

specification dg_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (dgValues M s DHPrev DO O H0 U t T D BD) := by
  obtain ⟨final, he, hm⟩ := dg_memory M DHPrev DO O H0 DG U t T D BD s
  exact ⟨final, he, post_of_memory s final DG t T D BD _ hm⟩




#axiomsClean forward_fp32_stage
#axiomsClean dx_fp32_stage
#axiomsClean dx_two_fp32_stage
#axiomsClean dg_fp32_stage


end
end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
