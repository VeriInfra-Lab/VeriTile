import VeriTile.Triton.Float.IntegerDotPerAddressProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.FP.Structural.IntegerDot.Regression
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Deliberately non-associative primitives expose ordered carries. -/
def probe : Arithmetic where
  normalize := fun z => z % 256
  product8 := fun a b => a.toNat * b.toNat
  add := fun a b => a * 10 + b
  mma32 := fun isa a b carry =>
    carry * 100 + (a 0).toNat + (b 31).toNat +
      (match isa with | .mmaM16N8K32Satfinite => 1 | .wgmmaM64K32 _ => 2)

def scalarPlan (k : Nat) (fused : Bool) : Plan k where
  operations := (List.finRange k).map Micro.scalar
  complete := by
    simpa only [List.flatMap_map,Function.comp_def,Micro.terms,List.flatMap_singleton'] using
      List.Perm.refl (List.finRange k)
  fused := fused

theorem arbitrary_carry_is_normalized :
    (scalarPlan 0 true).evalInt probe (fun i => Fin.elim0 i) (fun i => Fin.elim0 i) 258 = 2 := by
  rfl

theorem nonfused_preserves_final_add :
    (scalarPlan 0 false).evalInt probe (fun i => Fin.elim0 i) (fun i => Fin.elim0 i) 258 = 2580 := by
  rfl

def reversePlan : Plan 2 where
  operations := [.scalar 1,.scalar 0]
  complete := by decide
  fused := true

theorem finite_order_is_observable :
    (scalarPlan 2 true).evalInt probe (fun i => BitVec.ofNat 8 (i.val+1)) (fun _ => 1) 2 = 212 ∧
    reversePlan.evalInt probe (fun i => BitVec.ofNat 8 (i.val+1)) (fun _ => 1) 2 = 221 := by
  constructor <;> rfl

def holeLanes (i : Fin 32) : Option (Fin 1) := if i = 0 then some 0 else none
def instructionPlan (isa : ISA) : Plan 1 where
  operations := [.instruction isa holeLanes]
  complete := by
    change ((List.ofFn holeLanes).filterMap id).Perm (List.finRange 1)
    decide
  fused := true

theorem instruction_tag_and_zero_holes :
    (instructionPlan .mmaM16N8K32Satfinite).evalInt probe (fun _ => 7) (fun _ => 11) 258 = 208 ∧
    (instructionPlan (.wgmmaM64K32 .n64)).evalInt probe (fun _ => 7) (fun _ => 11) 258 = 209 := by
  constructor <;> rfl

def context : Context where
  rows := 1
  cols := 1
  rowRegister := "row"
  columnRegister := "column"
  iterationRegister := "iteration"
  splitAxis := 1
  plans := fun k _ _ _ iteration => scalarPlan k (decide (iteration = 0))

def accumulate : ComputeStmt :=
  .assign .int [1,1] "acc" (.alg (.add .int (.consSame (.consSame .nil))
    (.ref .int [1,1] "acc")
    (.dotInt (batch := []) (.full [1,0] (.constInt 7)) (.full [0,1] (.constInt 11)) {})))

def begin (s : State α) : State α := s.setReg "acc" .int [1,1] (fun _ => 258)
def first (s : State α) : State α :=
  ((begin s).setReg "iteration" .nat [] (fun _ => 0)).setReg "acc" .int [1,1] (fun _ => 2)
def second (s : State α) : State α :=
  ((first s).setReg "iteration" .nat [] (fun _ => 1)).setReg "acc" .int [1,1] (fun _ => 20)

theorem loop_uses_actual_context {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    range M probe context "iteration" 0 2 1 [accumulate] (begin s) = some (second s) := by
  simp [range_unfold,run_unfold,step_unfold,arithmeticStep,addExpression,
    accumulate,evalOp_unfold,State.setReg,begin,first,second,dotAdd,context,
    scalarPlan,Plan.evalInt,probe,Broadcast.leftIndex,List.finRange_zero]

/-- The fused node consumes the actual source children, including the INT8
operand conversion, rather than a reference result supplied by the context. -/
theorem actual_source_operands {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    addExpression M probe context (.scalarL : Broadcast [] [1,1] [1,1]) (.constInt 258)
      (.dotInt (batch := []) (.full [1,1] (.constInt 263)) (.full [1,1] (.constInt 11)) {})
      (s.setReg "iteration" .nat [] (fun _ => 0)) = some (fun _ => 97) := by
  simp [addExpression,evalOp_unfold,State.setReg,dotAdd,context,scalarPlan,
    Plan.evalInt,Micro.eval,Arithmetic.instructions,probe,Broadcast.leftIndex,List.finRange_succ]

def writeAtCarry : ComputeStmt := .store .int [1,1]
  (.region "out" (.castIntToNat (.ref .int [1,1] "acc")))
  (.alg (.ref .int [1,1] "acc")) .none

/-- Safety uses the value produced by the second actual iteration. -/
theorem successor_address_exact {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) :
    stepSafe M probe context bounds writeAtCarry (second s) ↔ 20 < bounds "out" := by
  simp [stepSafe_unfold,writeAtCarry,Safety.stepSafe_unfold,Safety.safeAddress,
    Safety.safeExpr,Safety.safeMask,Safety.safeOp_unfold,Safety.addressSafe,
    Safety.maskActive,evalOp_unfold,second,State.setReg,Region.cast]

theorem unsupported_effect_rejected {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (bounds : RegionName → Nat) :
    ¬ stepSafe M probe context bounds (.effectMarker "unknown") s := by
  simp [stepSafe_unfold,Safety.stepSafe_unfold]

theorem untaken_branch_has_no_access {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) :
    stepSafe M probe context (fun _ => 0)
      (.ifThenElse (.constBool false) [writeAtCarry] []) (second s) := by
  simp [stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold]

theorem missing_source_input_fails {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (missing : s.regs .int [1,1] "acc" = none) :
    step M probe context accumulate s = none := by
  simp [step_unfold,arithmeticStep,addExpression,accumulate,evalOp_unfold,missing,context]

theorem invocation_uses_same_source_safety {α : Type} [Inhabited α]
    (job : IntegerDotInvocation α) (s : State α) (bounds : RegionName → Nat) :
    job.Safe bounds s ↔ TraceSafe job.backend job.arithmetic job.context bounds
      job.kernel (job.toInvocation.initial s) := Iff.rfl

#axiomsClean arbitrary_carry_is_normalized
#axiomsClean nonfused_preserves_final_add
#axiomsClean finite_order_is_observable
#axiomsClean instruction_tag_and_zero_holes
#axiomsClean loop_uses_actual_context
#axiomsClean actual_source_operands
#axiomsClean successor_address_exact
#axiomsClean unsupported_effect_rejected
#axiomsClean untaken_branch_has_no_access
#axiomsClean missing_source_input_fails
#axiomsClean invocation_uses_same_source_safety
end
end VeriTile.Triton.FP.Structural.IntegerDot.Regression
