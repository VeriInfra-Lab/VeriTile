import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Output
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IndexPair
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual block-pointer and masked scalar addresses. Public input/output
windows will establish these bounds for each complete source launch. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  query : ∀ i : TileIndex [128,p.DIM], s.pids 0*128+i.1.val < p.SEQLEN →
    base p s p.stride_q_bs p.stride_q_head+
      (s.pids 0*128+i.1.val)*p.stride_q_seqlen+i.2.1.val*p.stride_q_dim < bounds "Q"
  key : ∀ row (i : TileIndex [p.DIM,64]), row+i.2.1.val < p.SEQLEN →
    base p s p.stride_k_bs p.stride_k_head+i.1.val*p.stride_k_dim+
      (row+i.2.1.val)*p.stride_k_seqlen < bounds "K"
  value : ∀ row (i : TileIndex [64,p.DIM]), row+i.1.val < p.SEQLEN →
    base p s p.stride_v_bs p.stride_v_head+(row+i.1.val)*p.stride_v_seqlen+
      i.2.1.val*p.stride_v_dim < bounds "V"
  output : ∀ i : TileIndex [128,p.DIM], active p s i → address p s i < bounds "O"
  log : ∀ i : TileIndex [128], lActive p s i → lAddress p s i < bounds "L"

theorem traceSafe_singleton (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

theorem assign_keeps_reg (M : Algebra α) (dtype shape name) (expr : ComputeExpr dtype shape)
    (s t : State α) (other : RegName) (different : other ≠ name)
    (d : TileDType) (sh : TileShape)
    (h : step M (.assign dtype shape name expr) s = some t) :
    t.regs d sh other=s.regs d sh other := by
  simp only [step] at h
  cases he : evalExpr M expr s with
  | none => simp [he] at h
  | some value =>
    simp only [he,bind,pure,Option.bind_some,Option.some.injEq] at h
    subst t
    simp [State.setReg,different]

theorem traceSafe_of_pointwise (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (hs : ∀ st ∈ code, ∀ s : State α, Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (hs st (by simp) s)
    intro t _
    exact ih (fun st h => hs st (by simp [h])) t

theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem safeAdvance2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (shape : TileShape)
    (pointer : Op .blockPtr shape) (a b : Op .int []) :
    Safety.safeOp M precision bounds s (.advanceBlockPtrDyn pointer [a,b]) ↔
      Safety.safeOp M precision bounds s pointer ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

/-- The source prefix through the Q block-pointer creation; no memory access occurs
until the following Q load. -/
def preQuery (p : Parameters) : List ComputeStmt := (parts p).before.take 9

theorem preQuery_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (preQuery p) s := by
  let pointer : ComputeStmt := ((kernel p).surfaceBody[8]?).getD (.effectMarker "tl.debug_barrier")
  have split : preQuery p =
      FlashAttnScalarABIChecked.IndexPair.prelude Bool.false p ++ [pointer] := by rfl
  rw [split,Safety.traceSafeList_append]
  refine ⟨FlashAttnScalarABIChecked.IndexPair.prelude_safe _ _ _ _ _,?_⟩
  intro t _
  rw [traceSafe_singleton]
  change Safety.stepSafe (engine (withDot M dot)) bounds
    (.assign .blockPtr [128,p.DIM] "Q_block_ptr"
      (.alg (.makeBlockPtrDynOffsets ⟨"Q"⟩ (.ref .nat [] "q_base_offset")
        [p.SEQLEN,p.DIM] [128,p.DIM] [p.stride_q_seqlen,p.stride_q_dim]
        [.mul .nat .nil (.ref .nat [] "start_m") (.constNat 128),.constNat 0]))) t
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,safeBlockPtr2]
  simp [Safety.safeOp_unfold]

#axiomsClean preQuery_safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
