import bench.tritonbench_g.fused_recurrent_delta.FusedRecurrentDelta
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages

/-! Typed fp32 delta-rule stage contracts. Materialized HPrev/HNext carry
tiles retain the previously accepted stage boundary. Arithmetic, casts and
reductions are calls to the selected floating backend, with no real-ring
identity or assumed final result. Reduction order is an explicit backend
choice; a native specialization must bind it to the actual lowering. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

def output (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat)
    (scale : ℝ) : ComputeKernel :=
  ComputeKernel.mk [HNext, q] [o] (triton {
  tl.region HNext = tl.float32, q = tl.float32, o = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HNext + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  _o = prev * b_q[None, :]
  _o = tl.sum(_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (_o).to(o.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def keyIndex {α : Type} (s : State α) (BK : Nat) (j : Fin BK) : Nat := s.pids 1 * BK + j.val
def valueIndex {α : Type} (s : State α) (BV : Nat) (i : Fin BV) : Nat := s.pids 0 * BV + i.val
def stateAddress {α : Type} (s : State α) (K V BK BV : Nat) (i : TileIndex [BV,BK]) : Nat :=
  s.pids 2 * K * V + keyIndex s BK i.2.1 * V + valueIndex s BV i.1
def keyAddress {α : Type} (s : State α) (t stride K BK : Nat) (i : TileIndex [BK]) : Nat :=
  s.pids 2 * stride + s.pids 1 * BK + i.1.val + t * K
def valueAddress {α : Type} (s : State α) (t stride V BV : Nat) (i : TileIndex [BV]) : Nat :=
  s.pids 2 * stride + s.pids 0 * BV + i.1.val + t * V
def outputAddress {α : Type} (s : State α) (t stride B H V BV : Nat) (i : TileIndex [BV]) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * stride + s.pids 0 * BV + i.1.val + t * V

def maskedInput {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (off : Nat) (active : Prop) [Decidable active] : α :=
  if active then (s.mem R off).read .real else M.literal (some .fp32) .real 0

/-- Separate scalar/product graph and opaque reduction call; no sum
reassociation, mathematical finite sum, or zero-annihilator law is used. -/
def outputValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HNext q : RegionName) (t stride K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α :=
  let products : TileIndex [BV,BK] → α := fun i =>
    M.binary (some .fp32) .real .mul
      (M.fp32Load (maskedInput M s HNext (stateAddress s K V BK BV i)
        (keyIndex s BK i.2.1 < K ∧ valueIndex s BV i.1 < V)))
      (M.binary (some .fp32) .real .mul
        (M.cast (some .fp32) .real .real
          (maskedInput M s q (keyAddress s t stride K BK i.2) (keyIndex s BK i.2.1 < K)))
        (M.literal (some .fp32) .real scale))
  let sums := M.reduceSum (some .fp32) (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false products
  fun i => M.cast (some .fp32) .real .real (sums i)

theorem output_memory {α : Type} [Inhabited α] (M : Algebra α)
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (output HNext q o t s_qk_h s_vo_h B H K V BK BV scale) s = some final ∧
      final.mem = (scatter s o BV (outputAddress s t s_vo_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s HNext q t s_qk_h K V BK BV scale)).mem := by
  simp [output, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, outputValues, maskedInput,
    scatter, outputAddress, stateAddress, keyAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem outputAddress_injective {α : Type} (s : State α) (t stride B H V BV : Nat) :
    Function.Injective (outputAddress s t stride B H V BV) := by
  intro i j h
  have he : i.1 = j.1 := by
    apply Fin.ext
    dsimp [outputAddress] at h
    omega
  exact Prod.ext he (Subsingleton.elim _ _)

def Post {α : Type} (s final : State α) (o : RegionName) (BV : Nat)
    (address : TileIndex [BV] → Nat) (active : TileIndex [BV] → Prop)
    (values : TileIndex [BV] → α) : Prop :=
  (∀ i, active i → final.mem o (address i) = .mk .real (values i)) ∧
  (∀ R off, (R ≠ o ∨ ∀ i, active i → off ≠ address i) → final.mem R off = s.mem R off)

theorem post_of_memory {α : Type} (s final : State α) (o : RegionName) (BV : Nat)
    (address : TileIndex [BV] → Nat) (active : TileIndex [BV] → Prop) [DecidablePred active]
    (values : TileIndex [BV] → α) (hinj : Function.Injective address)
    (hm : final.mem = (scatter s o BV address active values).mem) :
    Post s final o BV address active values := by
  constructor
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback o address (fun j => .mk .real (values j)) active s i hi hinj
  · intro R off h
    rw [hm]
    apply State.masked_scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => h i hi

specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (output HNext q o t s_qk_h s_vo_h B H K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s HNext q t s_qk_h K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := output_memory M HNext q o t s_qk_h s_vo_h B H K V BK BV scale s
  exact ⟨final, he, post_of_memory s final o BV _ _ _ (outputAddress_injective s t s_vo_h B H V BV) hm⟩


/-- The original in-place v_new stage, with fp32 load conversions retained. -/
def vnew (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) : ComputeKernel :=
  ComputeKernel.mk [HPrev,k,v] [v] (triton {
  tl.region HPrev = tl.float32, k = tl.float32, v = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  tl.store(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    (b_v).to(v.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def readoutValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev k : RegionName) (t stride K V BK BV : Nat) : TileIndex [BV] → α :=
  let products : TileIndex [BV,BK] → α := fun i =>
    M.binary (some .fp32) .real .mul
      (M.fp32Load (maskedInput M s HPrev (stateAddress s K V BK BV i)
        (keyIndex s BK i.2.1 < K ∧ valueIndex s BV i.1 < V)))
      (M.cast (some .fp32) .real .real
        (maskedInput M s k (keyAddress s t stride K BK i.2) (keyIndex s BK i.2.1 < K)))
  M.reduceSum (some .fp32) (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false products

def residualValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) : TileIndex [BV] → α := fun i =>
  M.binary (some .fp32) .real .sub
    (M.cast (some .fp32) .real .real
      (maskedInput M s v (valueAddress s t s_vo_h V BV i) (valueIndex s BV i.1 < V)))
    (readoutValues M s HPrev k t s_qk_h K V BK BV i)

def vnewValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) : TileIndex [BV] → α := fun i =>
  M.cast (some .fp32) .real .real (residualValues M s HPrev k v t s_qk_h s_vo_h K V BK BV i)

theorem vnew_memory {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      final.mem = (scatter s v BV (valueAddress s t s_vo_h V BV)
        (fun i => valueIndex s BV i.1 < V) (vnewValues M s HPrev k v t s_qk_h s_vo_h K V BK BV)).mem := by
  simp [vnew, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, vnewValues, residualValues, readoutValues, maskedInput,
    scatter, valueAddress, stateAddress, keyAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

theorem valueAddress_injective {α : Type} (s : State α) (t stride V BV : Nat) :
    Function.Injective (valueAddress s t stride V BV) := by
  intro i j h
  have he : i.1 = j.1 := by
    apply Fin.ext
    dsimp [valueAddress] at h
    omega
  exact Prod.ext he (Subsingleton.elim _ _)

specification vnew_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV)
        (fun i => valueIndex s BV i.1 < V) (vnewValues M s HPrev k v t s_qk_h s_vo_h K V BK BV) := by
  obtain ⟨final, he, hm⟩ := vnew_memory M HPrev k v t s_qk_h s_vo_h K V BK BV s
  exact ⟨final, he, post_of_memory s final v BV _ _ _ (valueAddress_injective s t s_vo_h V BV) hm⟩

#axiomsClean vnew_fp32_stage

#axiomsClean output_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
