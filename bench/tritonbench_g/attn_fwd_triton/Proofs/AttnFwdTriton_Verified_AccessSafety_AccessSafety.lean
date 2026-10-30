import bench.tritonbench_g.attn_fwd_triton.Proofs.Output
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealAccessSafety

/-! Local active-address bounds for the exact repaired masked loads/store.
No bound is demanded for inactive pointer lanes. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.AccessSafety
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def QueryBound (bounds : RegionBounds) (c : Config) (BM BD : Nat) : Prop :=
  ∀ (r : Fin BM) (d : Fin BD), c.startM*BM+r.val < c.p.context → d.val < c.p.headDim →
    qbase c+(c.startM*BM+r.val)*c.p.queryRow+d.val*c.p.queryDim < bounds c.p.Q
def KeyBound (bounds : RegionBounds) (c : Config) (BN BD start : Nat) : Prop :=
  ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.p.context → d.val < c.p.headDim →
    kbase c+d.val*c.p.keyDim+(start+j.val)*c.p.keyRow < bounds c.p.K
def ValueBound (bounds : RegionBounds) (c : Config) (BN BD start : Nat) : Prop :=
  ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.p.context → d.val < c.p.headDim →
    vbase c+(start+j.val)*c.p.valueRow+d.val*c.p.valueDim < bounds c.p.V
def OutputBound (bounds : RegionBounds) (c : Config) (BM BD : Nat) : Prop :=
  ∀ i, Output.active c BM BD i = Bool.true → Output.address c BM BD i < bounds c.p.Out

theorem zero_load_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape) (mask : Op .bool shape) (bits : Tile .bool shape)
    (hp : s.regs .ptr shape name = some ptrs) (hm : evalOp mask s = some bits)
    (hs : mask.SafeAt bounds s)
    (hb : ∀ i, bits.data i = Bool.true → (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (Op.load .real (.ptr (.ref .ptr shape name)) (.maskOther mask (.broadcast (.const 0.0) shape))).SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  exact ⟨by simp [Op.SafeAt.eq_def],⟨hs,by simp [Op.SafeAt.eq_def]⟩,
    RealAccessSafety.masked_address_safe bounds s name ptrs mask bits hp hm hb⟩

theorem query_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BD : Nat)
    (hp : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hb : QueryBound bounds c BM BD) :
    (Op.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs"))
      (.maskOther (queryMask BM BD c.p.headDim) (.broadcast (.const 0.0) [BM,BD]))).SafeAt bounds s := by
  apply zero_load_safe bounds s "Q_ptrs" _ _ _ hp (eval_query_mask s BM BD _ _ _ hr hctx)
    (by simp [queryMask,Op.SafeAt.eq_def])
  rintro ⟨r,d,⟨⟩⟩ hi
  have h : c.startM*BM+r.val < c.p.context ∧ d.val < c.p.headDim := by
    simpa only [queryMaskTile,decide_eq_true_eq] using hi
  exact hb r d h.1 h.2

theorem key_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (name : RegName) (hp : s.regs .ptr [BD,BN] name = some (keyPointer c BD BN start))
    (hm : s.regs .bool [BD,BN] "k_mask" = some (keyMaskTile BD BN c.p.context c.p.headDim start))
    (hb : KeyBound bounds c BN BD start) :
    (Op.load .real (.ptr (.ref .ptr [BD,BN] name))
      (.maskOther (.ref .bool [BD,BN] "k_mask") (.broadcast (.const 0.0) [BD,BN]))).SafeAt bounds s := by
  apply zero_load_safe bounds s name _ _ _ hp (by simpa only [evalOp_ref] using hm)
    (by simp [Op.SafeAt.eq_def])
  rintro ⟨d,j,⟨⟩⟩ hi
  have h : start+j.val < c.p.context ∧ d.val < c.p.headDim := by
    simpa only [keyMaskTile,decide_eq_true_eq] using hi
  exact hb j d h.1 h.2

theorem value_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (second : Bool) (hp : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hb : ValueBound bounds c BN BD start) :
    Stmt.TraceSafe bounds (valueLoad BN BD c.p.headDim second) s := by
  rw [valueLoad,Stmt.TraceSafe]
  apply zero_load_safe bounds s (valueName second) _ _ _ hp (eval_value_mask s BN BD _ _ _ hn hs hctx)
    (by simp [valueMask,tail,Op.SafeAt.eq_def])
  rintro ⟨j,d,⟨⟩⟩ hi
  have h : start+j.val < c.p.context ∧ d.val < c.p.headDim := by
    simpa only [valueMaskTile,decide_eq_true_eq] using hi
  exact hb j d h.1 h.2

theorem scale_safe (bounds : RegionBounds) (s : BlockState) (region : RegionName) (offset : Nat) (name : RegName)
    (hp : s.regs .ptr [] name = some (Tile.scalar (region.cast,offset))) (hb : offset < bounds region) :
    (Op.load .real (.ptr (.ref .ptr [] name)) .none).SafeAt bounds s :=
  pointer_load_safe bounds s name _ hp (fun _ => hb)

theorem output_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BD : Nat)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hb : OutputBound bounds c BM BD) :
    Stmt.TraceSafe bounds (Output.store BM BD c.p.headDim) s := by
  rw [Output.store,Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],
    by simp [MaskOpt.SafeAt,queryMask,Op.SafeAt.eq_def],?_⟩
  exact RealAccessSafety.masked_address_safe bounds s "O_block_ptr" _ _ _ hp
    (eval_query_mask s BM BD _ _ _ hr hctx) hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.AccessSafety
