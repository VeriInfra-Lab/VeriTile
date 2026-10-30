import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeCarrier

namespace VeriTile.Bench.Repairs.DecayCumsum.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

theorem prepare_q_lowering (p : Instructions) (scale : Word) (q g : Node) :
    FPReference.prepareQ (algebra p selected .prepare scale) scaleSlot q g=
      atom (prepareQWord p scale q.value g.value) := by
  simp [FPReference.prepareQ,FPReference.mul,FPReference.exp2,FPReference.gate,
    algebra,multiply,load,cast,exponential,evalSite,selected,Expr.eval,arguments,prepareQWord]

theorem prepare_k_lowering (p : Instructions) (scale : Word) (k last g : Node) :
    FPReference.prepareK (algebra p selected .prepare scale) k last g=
      atom (prepareKWord p k.value last.value g.value) := by
  simp [FPReference.prepareK,FPReference.mul,FPReference.exp2,FPReference.gate,FPReference.sub,
    algebra,multiply,subtract,load,cast,exponential,evalSite,selected,Expr.eval,arguments,prepareKWord]

theorem backward_q_lowering (p : Instructions) (scale : Word) (inner inter g : Node) :
    FPReference.backwardQ (algebra p selected .backward scale) inner inter g=
      ⟨backwardQWord p inner.value inter.value g.value,.dq,none⟩ := by
  simp [FPReference.backwardQ,FPReference.add,FPReference.mul,FPReference.exp2,FPReference.gate,
    algebra,add,multiply,load,cast,exponential,evalSite,selected,Expr.eval,arguments,backwardQWord]

theorem backward_k_lowering (p : Instructions) (scale : Word) (inner inter last g : Node) :
    FPReference.backwardK (algebra p selected .backward scale) inner inter last g=
      ⟨backwardKWord p inner.value inter.value last.value g.value,.dk,none⟩ := by
  simp [FPReference.backwardK,FPReference.add,FPReference.mul,FPReference.exp2,FPReference.gate,FPReference.sub,
    algebra,add,multiply,subtract,load,cast,exponential,evalSite,selected,Expr.eval,arguments,backwardKWord]

theorem gradient_lowering (p : Instructions) (scale dq dk : Word) (q k : Node) :
    FPReference.gradient (algebra p selected .backward scale) ⟨dq,.dq,none⟩ ⟨dk,.dk,none⟩ q k=
      ⟨gradientWord p dq dk q.value k.value,.gradient,none⟩ := by
  simp [FPReference.gradient,FPReference.mul,FPReference.sub,
    algebra,multiply,subtract,load,evalSite,selected,Expr.eval,arguments,gradientWord]

theorem forward_lowering (p : Instructions) (scale : Word) (g : Nat → Node) (n : Nat) :
    FPReference.forwardCarry (algebra p selected .forward scale) g n=
      ⟨forwardWords p (fun t => (g t).value) n,if n=0 then .initialZero else .forwardCarry,none⟩ := by
  induction n with
  | zero => simp [FPReference.forwardCarry,FPReference.zero,algebra,forwardWords]
  | succ n ih =>
    simp only [FPReference.forwardCarry,ih]
    by_cases first : n=0
    all_goals simp [FPReference.add,FPReference.mul,FPReference.gate,algebra,
      add,multiply,cast,evalSite,selected,Expr.eval,arguments,forwardWords,first]

theorem backward_lowering (p : Instructions) (scale : Word) (span : Nat)
    (dqi dqo dki dko q k g : Nat → Node) (n : Nat) :
    FPReference.backwardCarry (algebra p selected .backward scale) span dqi dqo dki dko q k g n=
      ⟨backwardWords p span (fun t => (dqi t).value) (fun t => (dqo t).value)
        (fun t => (dki t).value) (fun t => (dko t).value) (fun t => (q t).value)
        (fun t => (k t).value) (fun t => (g t).value) n,
        if n=0 then .initialZero else .reverseCarry,none⟩ := by
  induction n with
  | zero => simp [FPReference.backwardCarry,FPReference.zero,algebra,backwardWords]
  | succ n ih =>
    simp only [FPReference.backwardCarry,ih,backward_q_lowering,backward_k_lowering,gradient_lowering]
    by_cases first : n=0
    all_goals simp [FPReference.add,algebra,add,evalSite,selected,Expr.eval,arguments,backwardWords,first]

theorem stored_lowering (p : Instructions) (kind : Kind) (scale : Word) (node : Node) :
    FPReference.stored (algebra p selected kind scale) node=storedWord node.value := rfl

/-- Complete per-lane graph bridge, including every iteration and store.
There is no split=fused numerical premise, including at the first step. -/
theorem evaluate_lowering (p : Instructions) (kind : Kind) (span : Nat) (scale : Word)
    (read : RegionName → Nat → Node) (r : RegionName) (t : Nat) :
    FPLogical.evaluate (algebra p selected kind scale) kind span scaleSlot read r t=
      storedWord (evaluateWords p kind span scale (fun s n => (read s n).value) r t) := by
  cases kind
  · simp [FPLogical.evaluate,FPReference.forward,forward_lowering,stored_lowering,evaluateWords]
  · simp only [FPLogical.evaluate,evaluateWords]
    split <;> simp [prepare_q_lowering,prepare_k_lowering,stored_lowering,atom]
  · simp only [FPLogical.evaluate,evaluateWords]
    split
    · simp [backward_q_lowering,stored_lowering]
    · split
      · simp [backward_k_lowering,stored_lowering]
      · simp [backward_lowering,stored_lowering]

theorem reference_lowering (p : Instructions) (h : Call) (kind : Kind) (scale : Word)
    (data : LogicalStorage.Data (Cell Node)) (r : RegionName) (c : Logical.Coordinate h) :
    FPLogical.reference (algebra p selected kind scale) h kind scaleSlot data r c=
      storedWord (referenceWords p h kind scale (fun s b t k => ((data s b t k).read .real).value) r c) := by
  exact evaluate_lowering p kind _ scale _ r _

#axiomsClean evaluate_lowering
#axiomsClean reference_lowering
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.Native
