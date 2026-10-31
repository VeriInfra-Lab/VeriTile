import VeriTile.Triton.Float.Structural

/-!
Actual-source access safety for the structural evaluator.

This is the structural counterpart of `Op.SafeAt` and `Stmt.TraceSafe` in
`Memory/Flatten.lean`. Addresses, masks, branches and successor states are
computed by the SAME `evalOp`, `step` and `run` used for the source execution.
No floating identities, compiler correspondence, or execution assumption is
introduced. A kernel IO theorem proves this predicate in addition to successful
execution and its output/frame contract.

Only active memory reads/writes require backing. Forming a pointer, including an
out-of-bounds pointer in a masked-off lane, is not a memory access. The predicate
is conservative for eager expression subterms; unsupported statement effects
and opaque expressions fail closed. It does not claim race freedom, physical
byte alignment, or a flat-pointer relocation theorem.
-/
namespace VeriTile.Triton.FP.Structural
namespace Safety
noncomputable section

/-- Precision selected by a source instruction that overrides its context. -/
def precisionContext (precision : Option IndexComputePrecision)
    (current : Option ComputeDType) : Option ComputeDType :=
  match precision with
  | none => current
  | some .fp32 => some .fp32
  | some .fp64 => some .fp64

/-- A memory lane is active only when its actual mask evaluates to true. -/
def maskActive {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) :
    MaskOpt dtype shape → TileIndex shape → Prop
  | .none => fun _ => True
  | .mask cond | .maskOther cond _ => fun i =>
      ∃ active, evalOp M p cond s = some active ∧ active i = true

/-- Bounds for the actual addresses of active lanes. Block-pointer boundary
checks suppress accesses exactly as in `evalOp`, `store` and `atomicAdd`. -/
def addressSafe {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (bounds : RegionName → Nat) (s : State α) :
    MemAccess dtype shape → (TileIndex shape → Prop) → Prop
  | .region r off, active =>
      ∀ offsets, evalOp M p off s = some offsets →
        ∀ i, active i → offsets i < bounds (Region.cast r)
  | .ptr ptr, active =>
      ∀ pointers, evalOp M p ptr s = some pointers →
        ∀ i, active i → (pointers i).2 < bounds (pointers i).1
  | .blockPtr ptr checkedAxes, active =>
      ∀ pointers, evalOp M p ptr s = some pointers →
        ∀ i, active i →
          (pointers i).inBounds (TileShape.indexToList shape i) checkedAxes = true →
          (pointers i).address (TileShape.indexToList shape i) < bounds (pointers i).region

set_option maxHeartbeats 1600000 in
/-- Every load in an expression has a backed active address in this state.
Expression children retain the source precision context. This definition also
checks masked fallback expressions because the evaluator evaluates them. -/
def safeOp {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .numPrograms _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .philox4x32 _ _ _ seed c0 c1 c2 c3 =>
      safeOp M p bounds s seed ∧ safeOp M p bounds s c0 ∧ safeOp M p bounds s c1 ∧ safeOp M p bounds s c2 ∧ safeOp M p bounds s c3
  | .uniform32 word => safeOp M p bounds s word
  | .rand _ seed offsets => safeOp M p bounds s seed ∧ safeOp M p bounds s offsets
  | .randTensor _ seed offsets => safeOp M p bounds s seed ∧ safeOp M p bounds s offsets
  | .broadcast e _ => safeOp M p bounds s e
  | .full _ e => safeOp M p bounds s e
  | .withPrecision _ precision e =>
      safeOp M (some (match precision with | .fp32 => .fp32 | .fp64 => .fp64)) bounds s e
  | .castFloat _ _ e => safeOp M p bounds s e
  | .castNatToInt e => safeOp M p bounds s e
  | .negFloat _ e => safeOp M p bounds s e
  | .absFloat _ e => safeOp M p bounds s e
  | .floatToInt precision _ _ e => safeOp M (precisionContext precision p) bounds s e
  | .floatToNat precision _ _ e => safeOp M (precisionContext precision p) bounds s e
  | .castIntToNat e => safeOp M p bounds s e
  | .castRealToInt8 e => safeOp M p bounds s e
  | .decodeFloat8E5 _ e => safeOp M p bounds s e
  | .add _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .sub _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .mul _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .div _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .floorDiv _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .mod _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .bitAnd _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .bitOr _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .bitXor _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .shiftLeft _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .shiftRight _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .exp a => safeOp M p bounds s a
  | .libdeviceExp a => safeOp M p bounds s a
  | .libdeviceLog a => safeOp M p bounds s a
  | .libdeviceExpm1 a => safeOp M p bounds s a
  | .libdeviceTanh a => safeOp M p bounds s a
  | .libdeviceLog1p a => safeOp M p bounds s a
  | .exp2 a => safeOp M p bounds s a
  | .log a => safeOp M p bounds s a
  | .log2 a => safeOp M p bounds s a
  | .sigmoid a => safeOp M p bounds s a
  | .sqrt a => safeOp M p bounds s a
  | .rsqrt a => safeOp M p bounds s a
  | .tanh a => safeOp M p bounds s a
  | .sin a => safeOp M p bounds s a
  | .cos a => safeOp M p bounds s a
  | .tan a => safeOp M p bounds s a
  | .atan a => safeOp M p bounds s a
  | .cosh a => safeOp M p bounds s a
  | .sinh a => safeOp M p bounds s a
  | .erf a => safeOp M p bounds s a
  | .minPosInf a => safeOp M p bounds s a
  | .lt _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .le _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .eq _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .gt _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .ge _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .ne _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .boolAnd _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .boolOr _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .boolNot a => safeOp M p bounds s a
  | .max2 _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .maximumFloat _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .minimumFloat _ _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .pow _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .libdevicePowi _ a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .where c a b => safeOp M p bounds s c ∧ safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .ite c a b => safeOp M p bounds s c ∧ safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .reduceMin _ _ a => safeOp M p bounds s a
  | .reduceMax _ _ a => safeOp M p bounds s a
  | .reduceMaxNat _ _ a => safeOp M p bounds s a
  | .reduceMaxInt _ _ a => safeOp M p bounds s a
  | .reduceSum _ _ a => safeOp M p bounds s a
  | .reduceSumBool _ _ a => safeOp M p bounds s a
  | .reduceSumTyped _ _ _ a => safeOp M p bounds s a
  | .reduceMaxTyped _ _ _ a => safeOp M p bounds s a
  | .scan _ _ _ a => safeOp M p bounds s a
  | .argMax _ a => safeOp M p bounds s a
  | .argMaxInt _ a => safeOp M p bounds s a
  | .argMaxTyped _ _ a precision => safeOp M (precisionContext precision p) bounds s a
  | .indexedMaxValue _ _ a => safeOp M p bounds s a
  | .argMin _ a => safeOp M p bounds s a
  | .sort _ a => safeOp M p bounds s a
  | .dot a b _ => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .dotAcc _ _ _ a b acc _ => safeOp M p bounds s a ∧ safeOp M p bounds s b ∧ safeOp M p bounds s acc
  | .dotTyped _ _ _ a b _ => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .dotInt a b _ => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .transpose a => safeOp M p bounds s a
  | .reshape _ a => safeOp M p bounds s a
  | .remap _ _ a => safeOp M p bounds s a
  | .join a b => safeOp M p bounds s a ∧ safeOp M p bounds s b
  | .split _ a => safeOp M p bounds s a
  | .expandDim _ a => safeOp M p bounds s a
  | .ptrBase _ => True
  | .ptrAdd _ ptr off => safeOp M p bounds s ptr ∧ safeOp M p bounds s off
  | .ptrSub _ ptr off => safeOp M p bounds s ptr ∧ safeOp M p bounds s off
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => safeOp M p bounds s base
  | .makeBlockPtrDynOffsets _ base _ _ _ offsets =>
      safeOp M p bounds s base ∧ ∀ off ∈ offsets, safeOp M p bounds s off
  | .makeBlockPtrDynMetadata _ base _ _ strides offsets =>
      safeOp M p bounds s base ∧ (∀ stride ∈ strides,safeOp M p bounds s stride) ∧
        ∀ off ∈ offsets,safeOp M p bounds s off
  | .advanceBlockPtr ptr _ => safeOp M p bounds s ptr
  | .advanceBlockPtrDyn ptr deltas =>
      safeOp M p bounds s ptr ∧ ∀ delta ∈ deltas, safeOp M p bounds s delta
  | .load _ mem mask padding =>
      (match mem with
        | .region _ off => safeOp M p bounds s off
        | .ptr ptr => safeOp M p bounds s ptr
        | .blockPtr ptr _ => safeOp M p bounds s ptr) ∧
      (match mask with
        | .none => True
        | .mask m => safeOp M p bounds s m
        | .maskOther m other => safeOp M p bounds s m ∧ safeOp M p bounds s other) ∧
      addressSafe M p bounds s mem (maskActive M p s mask)
  | .natToReal a => safeOp M p bounds s a
  | .intToReal a => safeOp M p bounds s a


/-- Safety of an address expression; pointer formation itself is unrestricted. -/
def safeAddress {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (bounds : RegionName → Nat) (s : State α) :
    MemAccess dtype shape → Prop
  | .region _ off => safeOp M p bounds s off
  | .ptr ptr | .blockPtr ptr _ => safeOp M p bounds s ptr

/-- All evaluated mask children, including explicit fallback values. -/
def safeMask {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (bounds : RegionName → Nat) (s : State α) :
    MaskOpt dtype shape → Prop
  | .none => True
  | .mask cond => safeOp M p bounds s cond
  | .maskOther cond other => safeOp M p bounds s cond ∧ safeOp M p bounds s other

/-- Compute operations retain the source-selected precision and load type.
Unsupported compute operations cannot establish an IO safety certificate. -/
def safeCompute {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) : ComputeOp dtype shape → Prop
  | .alg d e => safeOp M (some d) bounds s e
  | .const (dtype := .fp64) _ => False
  | .const (dtype := .fp32) _ | .const (dtype := .int32) _ |
      .const (dtype := .uint32) _ => True
  | .full _ e => safeCompute M bounds s e
  | .load .fp64 .. => False
  | .load .fp32 mem mask padding => safeOp M (some .fp32) bounds s (.load .real mem mask padding)
  | .load .int32 mem mask padding => safeOp M (some .int32) bounds s (.load .int mem mask padding)
  | .load .uint32 mem mask padding => safeOp M (some .uint32) bounds s (.load .nat mem mask padding)
  | .bitcast .int32 .fp32 _ e => safeCompute M bounds s e
  | .bitcast .uint32 .fp32 _ e => safeCompute M bounds s e
  | .bitcast .. => False

/-- Compute-only expression nodes are checked in their actual evaluator. -/
def safeExpr {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) : ComputeExpr dtype shape → Prop
  | .alg e => safeOp M none bounds s e
  | .compute e => safeCompute M bounds s e
  | .isFinite _ e => safeExpr M bounds s e
  | .fma _ _ _ a b c => safeExpr M bounds s a ∧ safeExpr M bounds s b ∧ safeExpr M bounds s c
  | .fmaTyped _ _ _ a b c => safeExpr M bounds s a ∧ safeExpr M bounds s b ∧ safeExpr M bounds s c
  | .opaque _ | .llrint _ => False

/-- The algorithm statement fragment actually accepted by `stepAlg`. -/
def safeAlg {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) : Stmt → State α → Prop
  | .assign _ _ _ e, s => safeOp M none bounds s e
  | .store _ _ mem value mask, s =>
      safeAddress M none bounds s mem ∧ safeOp M none bounds s value ∧
      safeMask M none bounds s mask ∧ addressSafe M none bounds s mem (maskActive M none s mask)
  | .atomicAdd _ _ mem value mask, s =>
      safeAddress M none bounds s mem ∧ safeOp M none bounds s value ∧
      safeMask M none bounds s mask ∧ addressSafe M none bounds s mem (maskActive M none s mask)
  | _, _ => False

mutual
/-- Safety of one source statement at the state in which it executes.
Branches and loops use the actual evaluator; unsupported effects fail closed. -/
def stepSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) : ComputeStmt → State α → Prop
  | .alg st, s => safeAlg M bounds st s
  | .assign _ _ _ e, s => safeExpr M bounds s e
  | .store _ _ mem value mask, s =>
      safeAddress M none bounds s mem ∧ safeExpr M bounds s value ∧
      safeMask M none bounds s mask ∧ addressSafe M none bounds s mem (maskActive M none s mask)
  | .atomicAdd _ _ mem value mask, s =>
      safeAddress M none bounds s mem ∧ safeExpr M bounds s value ∧
      safeMask M none bounds s mask ∧ addressSafe M none bounds s mem (maskActive M none s mask)
  | .forLoop idx n body, s => loopSafe M bounds idx 0 n body s
  | .forRange idx start stop stride body, s => rangeSafe M bounds idx start stop stride body s
  | .forRangeDyn idx start stop stride body, s =>
      safeOp M none bounds s start ∧ safeOp M none bounds s stop ∧ safeOp M none bounds s stride ∧
      match evalOp M none start s, evalOp M none stop s, evalOp M none stride s with
      | some a, some b, some d => rangeSafe M bounds idx (a PUnit.unit) (b PUnit.unit) (d PUnit.unit) body s
      | _, _, _ => True
  | .forRangeInt idx start stop stride body, s =>
      safeOp M none bounds s start ∧ safeOp M none bounds s stop ∧ safeOp M none bounds s stride ∧
      match evalOp M none start s, evalOp M none stop s, evalOp M none stride s with
      | some a, some b, some d => rangeIntSafe M bounds idx (a PUnit.unit) (b PUnit.unit) (d PUnit.unit) body s
      | _, _, _ => True
  | .ifThen cond body, s =>
      safeOp M none bounds s cond ∧
      match evalOp M none cond s with
      | some c => if c PUnit.unit then traceSafeList M bounds body s else True
      | none => True
  | .ifThenElse cond yes no, s =>
      safeOp M none bounds s cond ∧
      match evalOp M none cond s with
      | some c => if c PUnit.unit then traceSafeList M bounds yes s else traceSafeList M bounds no s
      | none => True
  | .effectMarker "tl.debug_barrier", _ => True
  | _, _ => False
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

/-- Each statement is checked in its actual predecessor state. A failed step
has no successor; public IO additionally proves a successful execution. -/
def traceSafeList {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) : List ComputeStmt → State α → Prop
  | [], _ => True
  | st :: rest, s => stepSafe M bounds st s ∧
      match step M st s with
      | some t => traceSafeList M bounds rest t
      | none => True
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

/-- Counted-loop safety follows the existing loop's exact index sequence. -/
def loopSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (start stop : Nat)
    (body : List ComputeStmt) (s : State α) : Prop :=
  if start < stop then
    traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => start)) ∧
    match run M body (s.setReg idx .nat [] (fun _ => start)) with
    | some t => loopSafe M bounds idx (start + 1) stop body t
    | none => True
  else True
termination_by (sizeOf body + 1, stop - start)
decreasing_by all_goals omega

/-- A zero stride or empty natural interval makes no body access. -/
def rangeSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (cur stop stride : Nat)
    (body : List ComputeStmt) (s : State α) : Prop :=
  if stride = 0 then True else if cur < stop then
    traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => cur)) ∧
    match run M body (s.setReg idx .nat [] (fun _ => cur)) with
    | some t => rangeSafe M bounds idx (cur + stride) stop stride body t
    | none => True
  else True
termination_by (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega

/-- Signed dynamic ranges retain their signed indices and actual successor states. -/
def rangeIntSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (cur stop : Int) (stride : Nat)
    (body : List ComputeStmt) (s : State α) : Prop :=
  if stride = 0 then True else if cur < stop then
    traceSafeList M bounds body (s.setReg idx .int [] (fun _ => cur)) ∧
    match run M body (s.setReg idx .int [] (fun _ => cur)) with
    | some t => rangeIntSafe M bounds idx (cur + stride) stop stride body t
    | none => True
  else True
termination_by (sizeOf body + 1, (stop - cur).toNat)
decreasing_by all_goals omega
end

/- Materialize unfolding equations here. Downstream task proofs must not own
duplicate lazy equation declarations for the shared checker. -/
set_option maxHeartbeats 1600000 in
abbrev safeOp_unfold := @safeOp.eq_def
abbrev safeCompute_unfold := @safeCompute.eq_def
abbrev safeExpr_unfold := @safeExpr.eq_def
abbrev safeAddress_unfold := @safeAddress.eq_def
abbrev safeMask_unfold := @safeMask.eq_def
abbrev addressSafe_unfold := @addressSafe.eq_def
abbrev maskActive_unfold := @maskActive.eq_def
abbrev safeAlg_unfold := @safeAlg.eq_def
abbrev stepSafe_unfold := @stepSafe.eq_def
abbrev traceSafeList_unfold := @traceSafeList.eq_def
abbrev loopSafe_unfold := @loopSafe.eq_def
abbrev rangeSafe_unfold := @rangeSafe.eq_def
abbrev rangeIntSafe_unfold := @rangeIntSafe.eq_def

@[simp] theorem traceSafeList_nil {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) : traceSafeList M bounds [] s := by
  rw [traceSafeList]
  trivial

theorem traceSafeList_cons {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    traceSafeList M bounds (st :: rest) s ↔ stepSafe M bounds st s ∧
      ∀ t, step M st s = some t → traceSafeList M bounds rest t := by
  simp only [traceSafeList]
  cases h : step M st s <;> simp

theorem traceSafeList_cons_intro {α : Type} [Inhabited α] {M : Algebra α}
    {bounds : RegionName → Nat} {st : ComputeStmt} {rest : List ComputeStmt} {s : State α}
    (head : stepSafe M bounds st s)
    (tail : ∀ t, step M st s = some t → traceSafeList M bounds rest t) :
    traceSafeList M bounds (st :: rest) s :=
  (traceSafeList_cons M bounds st rest s).mpr ⟨head, tail⟩

/-- Prefix/suffix composition shares the actual intermediate execution state. -/
theorem traceSafeList_append {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (before after : List ComputeStmt) (s : State α) :
    traceSafeList M bounds (before ++ after) s ↔
      traceSafeList M bounds before s ∧
      ∀ t, run M before s = some t → traceSafeList M bounds after t := by
  induction before generalizing s with
  | nil => simp [run]
  | cons st rest ih =>
      cases h : step M st s <;>
        simp [traceSafeList, run, h, ih, and_assoc]

/-- Checked straight-line introduction with a known actual successor. -/
theorem traceSafeList_cons_of_step {α : Type} [Inhabited α] {M : Algebra α}
    {bounds : RegionName → Nat} {st : ComputeStmt} {rest : List ComputeStmt} {s t : State α}
    (head : stepSafe M bounds st s) (executed : step M st s = some t)
    (tail : traceSafeList M bounds rest t) : traceSafeList M bounds (st :: rest) s := by
  rw [traceSafeList, executed]
  exact ⟨head, tail⟩

end
end Safety

/-- The actual kernel source has only backed active memory accesses along its
structural execution. Successful execution, output values and frame are separate
obligations of a complete IO theorem. -/
def TraceSafe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (kernel : ComputeKernel) (initial : State α) : Prop :=
  Safety.traceSafeList M bounds kernel.surfaceBody initial

end VeriTile.Triton.FP.Structural
