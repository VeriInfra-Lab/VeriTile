import bench.tritonbench_g.int_scaled_matmul.IO
import bench.tritonbench_g.int_scaled_matmul.Proofs.PureSafety

namespace VeriTile.Bench.Repairs.IntScaledMatmul.BlockSafety
open VeriTile.Triton FP.Structural Revision001 BlockWord
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem safe_block_pair (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (r : RegionName)
    (base : Op .nat []) (parent : List Nat) (shape : TileShape) (strides : List Nat)
    (row col : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets r base parent shape strides [row,col]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s row ∧
        Safety.safeOp M precision bounds s col := by
  rw [Safety.safeOp_unfold]
  simp

attribute [local simp 2000] safe_block_pair

private theorem singleton_safe (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

/-- Pointer formation in the actual source prelude performs no memory access. -/
theorem prelude_safe (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M) bounds (prelude p) s := by
  apply PureSafety.trace
  intro st member current
  dsimp only [prelude,original,int_scaled_matmul_matmul_surface,ComputeKernel.surfaceBody,List.take] at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
  all_goals simp [Safety.stepSafe,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Both evaluated block-pointer boundary checks select exactly their declared
finite strided input matrices, including the final partial K tile. -/
theorem body_safe (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (h : Invariant p initial k current)
    (ha : ∀ r t,r<p.rows → t<p.inner →
      r*p.inputRowStride+t*p.inputInnerStride<bounds p.input)
    (hb : ∀ t c,t<p.inner → c<p.columns →
      t*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight) :
    Safety.traceSafeList (backend M) bounds (body p)
      (current.setReg "k" .nat [] (fun _ => k*p.blockInner)) := by
  have tail : ∀ s : State α,Safety.traceSafeList (backend M) bounds ((body p).drop 2) s := by
    intro s
    apply PureSafety.trace
    intro st member current
    simp [body,originalBody,original,int_scaled_matmul_matmul_surface,widths,ComputeKernel.surfaceBody] at member
    rcases member with (rfl | rfl | rfl)
    all_goals simp [Safety.stepSafe,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  rw [← List.take_append_drop 2 (body p),Safety.traceSafeList_append]
  refine ⟨?_,fun s _=>tail s⟩
  simp [body,originalBody,original,int_scaled_matmul_matmul_surface,widths,
    ComputeKernel.surfaceBody,singleton_safe,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,
    Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,eval_intToSigned,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.aptrs,h.bptrs,h.acc,aPointer,bPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  constructor
  · intro r t hr ht
    simpa only [Nat.mul_comm] using ha _ _ hr ht
  · intro t c ht hc
    simpa only [Nat.mul_comm] using hb _ _ ht hc

/-- The original output block pointer uses both checked axes. -/
theorem epilogue_safe (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (h : Invariant p initial (steps p) current)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output) :
    Safety.traceSafeList (backend M) bounds (epilogue p) current := by
  simp [epilogue,original,int_scaled_matmul_matmul_surface,ComputeKernel.surfaceBody,
    singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,FP.Structural.step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.pidm,h.pidn,h.acc,backend,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  intro r c hr hc
  simpa only [Nat.mul_comm] using ho _ _ hr hc

/-- Safety follows the exact successful predecessor states already used by
the complete source proof; no loop invariant is added to the public domain. -/
theorem loop_safe (M : Algebra α) (p : Parameters) (positive : 0<p.blockInner)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (hk : k≤steps p) (h : Invariant p initial k current)
    (ha : ∀ r t,r<p.rows → t<p.inner →
      r*p.inputRowStride+t*p.inputInnerStride<bounds p.input)
    (hb : ∀ t c,t<p.inner → c<p.columns →
      t*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight) :
    Safety.rangeSafe (backend M) bounds "k" (k*p.blockInner) p.inner p.blockInner
      (body p) current := by
  have aux : ∀ n k (current : State α),steps p-k=n → k≤steps p →
      Invariant p initial k current →
      Safety.rangeSafe (backend M) bounds "k" (k*p.blockInner) p.inner p.blockInner
        (body p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases active : k<steps p
      · let next := bodyExecution M p initial current k h
        have hi := bodyInvariant M p initial current k h
        rw [Safety.rangeSafe,if_neg (by omega),if_pos ((before_stop p positive k).mp active),
          next.property]
        refine ⟨body_safe M p bounds initial current k h ha hb,?_⟩
        simpa only [Nat.add_mul,Nat.one_mul] using
          ih (steps p-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi
      · have stop : ¬ k*p.blockInner<p.inner := by
          rw [← before_stop p positive]
          exact active
        rw [Safety.rangeSafe,if_neg (by omega),if_neg stop]
        trivial
  exact aux _ _ _ rfl hk h

theorem complete (M : Algebra α) (p : Parameters) (positive : 0<p.blockInner)
    (bounds : RegionName → Nat)
    (ha : ∀ r t,r<p.rows → t<p.inner →
      r*p.inputRowStride+t*p.inputInnerStride<bounds p.input)
    (hb : ∀ t c,t<p.inner → c<p.columns →
      t*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output)
    (initial : State α) : TraceSafe (backend M) bounds (source p) initial := by
  let start := preludeExecution M p initial
  have startInv := preludeInvariant M p initial
  obtain ⟨mid,executed,midInv⟩ := loopExecution M p positive initial start.val 0 (by omega) startInv
  simp only [Nat.zero_mul] at executed
  change Safety.traceSafeList (backend M) bounds (source p).surfaceBody initial
  rw [source_decomposition,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Safety.traceSafeList_append]
    refine ⟨prelude_safe M p bounds initial,?_⟩
    intro s hs
    rw [start.property] at hs
    cases hs
    rw [singleton_safe]
    rw [Safety.stepSafe]
    simpa only [Nat.zero_mul] using
      loop_safe M p positive bounds initial start.val 0 (by omega) startInv ha hb
  · intro final hf
    rw [run_append,start.property] at hf
    simp only [Option.bind_some,run,step,executed] at hf
    cases hf
    exact epilogue_safe M p bounds initial mid midInv ho

#axiomsClean complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul.BlockSafety
