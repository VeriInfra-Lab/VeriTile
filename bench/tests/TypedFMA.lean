import VeriTile.Triton.Float.IntegerStageSafety
import VeriTile.Meta.StatementAudit

/-! Typed FMA execution regressions. Symbolic operation traces distinguish one
fused primitive from two separately rounded operations without assuming any
floating numerical law. Loads retain the existing typed-cell read rule.
This file makes no compiler or GPU conformance claim. -/
namespace VeriTile.Tests.TypedFMA
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000

inductive Trace where
  | leaf (n : Nat)
  | fused (dtype : FloatDType) (a b c : Trace)
  | multiply (a b : Trace)
  | add (a b : Trace)
  deriving DecidableEq, Repr, Inhabited

def model (rest : Algebra Trace) : Algebra Trace :=
  { rest with
    fmaTyped := fun dtype => some (Trace.fused dtype)
    binary := fun _ _ op a b => match op with
      | .mul => .multiply a b
      | .add => .add a b
      | _ => rest.binary none .fp16 op a b }

def initial : State Trace where
  mem _ i := .mk .fp16 (.leaf i)
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef dtype _ _ := defaultValue dtype

def load (i : Nat) : ComputeExpr .fp16 [] :=
  .alg (.load .fp16 (.region ⟨"input"⟩ (.constNat i)) .none)

def fused (a b c : Nat) : ComputeExpr .fp16 [] :=
  .fmaTyped .fp16 .nil .nil (load a) (load b) (load c)

/-- Half operands reach one half primitive; no FP32 result or intermediate
mul/add operation is introduced. -/
theorem single_half_primitive (rest : Algebra Trace) :
    (evalExpr (model rest) (fused 0 1 2) initial).map (fun v => v PUnit.unit) =
      some (.fused .fp16 (.leaf 0) (.leaf 1) (.leaf 2)) := by
  simp [evalExpr,fused,load,model,evalOp_unfold,initial,Cell.read,
    ofFloat,toFloat,Region.cast,Broadcast.leftIndex,Broadcast.rightIndex]

def split : ComputeExpr .fp16 [] :=
  .alg (.add .fp16 .nil
    (.mul .fp16 .nil (.load .fp16 (.region ⟨"input"⟩ (.constNat 0)) .none)
      (.load .fp16 (.region ⟨"input"⟩ (.constNat 1)) .none))
    (.load .fp16 (.region ⟨"input"⟩ (.constNat 2)) .none))

/-- Replacing a fused primitive by separate mul/add changes the observable
operation trace. The arbitrary backend therefore permits no such rewrite. -/
theorem fused_not_split (rest : Algebra Trace) :
    (evalExpr (model rest) (fused 0 1 2) initial).map (fun v => v PUnit.unit) ≠
      (evalExpr (model rest) split initial).map (fun v => v PUnit.unit) := by
  rw [single_half_primitive]
  simp [split,evalExpr,evalOp_unfold,numeric,bop,model,initial,Cell.read,
    ofFloat,toFloat,Region.cast,Broadcast.leftIndex,Broadcast.rightIndex]

/-- The stored cell keeps the half tag as well as the fused result. -/
theorem store_keeps_half_tag (rest : Algebra Trace) :
    (step (model rest) (.store .fp16 [] (.region ⟨"out"⟩ (.constNat 0))
      (fused 0 1 2) .none) initial).map (fun s => s.mem "out" 0) =
        some (.mk .fp16 (.fused .fp16 (.leaf 0) (.leaf 1) (.leaf 2))) := by
  simp [step,evalExpr,fused,load,store,evalOp_unfold,model,initial,Cell.read,
    ofFloat,toFloat,Region.cast,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.allIndices,State.write]

/-- A differently tagged source cell still uses the old default-valued typed
read. Typed FMA does not reinterpret FP32 storage as a half payload. -/
theorem wrong_tag_uses_existing_read (rest : Algebra Trace) :
    (evalExpr (model rest) (fused 0 1 2)
      (initial.write "input" 2 (.mk .fp32 (.leaf 99)))).map (fun v => v PUnit.unit) =
      some (.fused .fp16 (.leaf 0) (.leaf 1) default) := by
  simp [evalExpr,fused,load,model,evalOp_unfold,initial,State.write,Cell.read,
    defaultValue,ofFloat,toFloat,Region.cast,Broadcast.leftIndex,Broadcast.rightIndex]

theorem missing_empty_fails {α : Type} [Inhabited α] (M : Algebra α)
    (state : State α) (a : ComputeExpr .fp16 [0]) (b c : ComputeExpr .fp16 [])
    (missing : M.fmaTyped .fp16 = none) :
    evalExpr M (.fmaTyped .fp16 .scalarR .scalarR a b c) state = none ∧
      QuantizeIntegerStage.evalExpr M none (fun _ => none)
        (.fmaTyped .fp16 .scalarR .scalarR a b c) state = none := by
  simp [evalExpr,QuantizeIntegerStage.evalExpr,missing]

/-- Both broadcast maps matter: a scalar times a vector is then broadcast
against a scalar addend, retaining a typed half output at every lane. -/
theorem broadcast_half {α : Type} [Inhabited α] (M : Algebra α) (state : State α)
    (n : Nat) (a c : ComputeExpr .fp16 []) (b : ComputeExpr .fp16 [n])
    (av cv : α) (bv : Values α .fp16 [n]) (f : α → α → α → α)
    (hf : M.fmaTyped .fp16 = some f)
    (ha : evalExpr M a state = some (fun _ => av))
    (hb : evalExpr M b state = some bv)
    (hc : evalExpr M c state = some (fun _ => cv)) :
    evalExpr M (.fmaTyped .fp16 .scalarL .scalarR a b c) state =
      some (fun i => f av (bv i) cv) := by
  simp [evalExpr,hf,ha,hb,hc,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex]

theorem integer_stage_keeps_half_primitive (rest : Algebra Trace) :
    (QuantizeIntegerStage.evalExpr (model rest) none (fun _ => none)
      (fused 0 1 2) initial).map (fun v => v PUnit.unit) =
        some (.fused .fp16 (.leaf 0) (.leaf 1) (.leaf 2)) := by
  simp [QuantizeIntegerStage.evalExpr,fused,load,model,evalOp_unfold,initial,Cell.read,
    ofFloat,toFloat,Region.cast,Broadcast.leftIndex,Broadcast.rightIndex]

/-- Mathematical projection keeps all typed children and both broadcasts. -/
theorem projection_typed (dtype : FloatDType) (bcAB : Broadcast a b ab)
    (bcC : Broadcast ab c out) (x : Op dtype.toTileDType a)
    (y : Op dtype.toTileDType b) (z : Op dtype.toTileDType c) :
    (ComputeExpr.fmaTyped dtype bcAB bcC (.alg x) (.alg y) (.alg z)).toAlgorithm? =
      .ok (.add dtype.numericDType bcC (.mul dtype.numericDType bcAB x y) z) := rfl

/-- The original FMA entry continues to dispatch through its original field. -/
theorem legacy_dispatch {α : Type} [Inhabited α] (M : Algebra α) (state : State α)
    (precision : IndexComputePrecision) (a b c : RegName)
    (newPrimitive : FloatDType → Option (α → α → α → α)) :
    evalExpr {M with fmaTyped := newPrimitive}
      (.fma precision .nil .nil (.alg (.ref .real [] a))
        (.alg (.ref .real [] b)) (.alg (.ref .real [] c))) state =
      evalExpr M (.fma precision .nil .nil (.alg (.ref .real [] a))
        (.alg (.ref .real [] b)) (.alg (.ref .real [] c))) state := rfl

theorem safety_checks_all_children {α : Type} [Inhabited α] (M : Algebra α)
    (state : State α) (bounds : RegionName → Nat)
    (a b c : ComputeExpr .fp16 []) :
    Safety.safeExpr M bounds state (.fmaTyped .fp16 .nil .nil a b c) ↔
      Safety.safeExpr M bounds state a ∧ Safety.safeExpr M bounds state b ∧
        Safety.safeExpr M bounds state c := Iff.rfl

theorem integer_stage_safety_checks_all_children {α : Type} [Inhabited α] (M : Algebra α)
    (state : State α) (bounds : RegionName → Nat) (bindings : QuantizeIntegerStage.RuntimeRegisters)
    (a b c : ComputeExpr .fp16 []) :
    QuantizeIntegerStage.Safety.safeExpr M bindings bounds state (.fmaTyped .fp16 .nil .nil a b c) ↔
      QuantizeIntegerStage.Safety.safeExpr M bindings bounds state a ∧
      QuantizeIntegerStage.Safety.safeExpr M bindings bounds state b ∧
      QuantizeIntegerStage.Safety.safeExpr M bindings bounds state c := Iff.rfl

def maskedStore (a b c : Nat) : ComputeStmt :=
  .store .fp16 [] (.region ⟨"out"⟩ (.constNat 0)) (fused a b c)
    (.mask (.constBool Bool.false))

/-- Even a disabled output store evaluates all three unmasked input loads.
An out-of-bounds read in any operand is therefore rejected by access safety. -/
theorem masked_output_still_checks_reads (M : Algebra Trace) :
    ¬ Safety.stepSafe M (fun _ => 1) (maskedStore 1 0 0) initial ∧
    ¬ Safety.stepSafe M (fun _ => 1) (maskedStore 0 1 0) initial ∧
    ¬ Safety.stepSafe M (fun _ => 1) (maskedStore 0 0 1) initial := by
  simp [maskedStore,fused,load,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,Region.cast]

#axiomsClean single_half_primitive
#axiomsClean fused_not_split
#axiomsClean store_keeps_half_tag
#axiomsClean wrong_tag_uses_existing_read
#axiomsClean missing_empty_fails
#axiomsClean broadcast_half
#axiomsClean integer_stage_keeps_half_primitive
#axiomsClean projection_typed
#axiomsClean legacy_dispatch
#axiomsClean masked_output_still_checks_reads
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.TypedFMA
