import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Native
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Native
noncomputable section
set_option maxHeartbeats 15000000
set_option maxRecDepth 20000
/-- Eager execution at each source store. The current-memory value is an
actual four-byte cell; register expressions remain available for FMA routing. -/
def materialize (p : Instructions) (policy : Policy) : Expr → Expr
  | .bits x => .bits x
  | .input a x => .input a x
  | .stored bytes => .stored bytes
  | .zero => .zero
  | .one => .one
  | .add a b => .add (materialize p policy a) (materialize p policy b)
  | .sub a b => .sub (materialize p policy a) (materialize p policy b)
  | .mul a b => .mul (materialize p policy a) (materialize p policy b)
  | .cut a => .stored (Codec.encode (a.eval p policy))
theorem materialize_code (p : Instructions) (policy : Policy) (e : Expr) :
    (materialize p policy e).code=e.code := by
  induction e <;> simp_all [materialize,Expr.code]
theorem materialize_value_aux (p : Instructions) (policy : Policy) (e : Expr) :
    ((materialize p policy e).eval p policy=e.eval p policy) ∧
    (∀ a b,e=Expr.mul a b →
      (materialize p policy a).eval p policy=a.eval p policy ∧
      (materialize p policy b).eval p policy=b.eval p policy) := by
  induction e with
  | bits x => simp [materialize,Expr.eval]
  | input a x => simp [materialize,Expr.eval]
  | stored bytes => simp [materialize,Expr.eval]
  | zero => simp [materialize,Expr.eval]
  | one => simp [materialize,Expr.eval]
  | cut a ih => simp [materialize,Expr.eval]
  | mul a b ha hb => simp [materialize,Expr.eval,ha.1,hb.1]
  | add a b ha hb =>
    cases a <;> cases b <;> simp_all [materialize,Expr.eval,Expr.code,materialize_code]
  | sub a b ha hb =>
    cases b <;> simp_all [materialize,Expr.eval,Expr.code,materialize_code]
theorem materialize_value (p : Instructions) (policy : Policy) (e : Expr) :
    (materialize p policy e).eval p policy=e.eval p policy := (materialize_value_aux p policy e).1

def storedOps (p : Instructions) (policy : Policy) : Ops Expr :=
  ⟨.zero,.one,.add,.sub,.mul,fun e => .stored (Codec.encode (e.eval p policy))⟩
theorem materialize_cut (p : Instructions) (policy : Policy) (e : Expr) :
    materialize p policy (graphOps.cut e)=(storedOps p policy).cut (materialize p policy e) := by
  simp [graphOps,storedOps,materialize,materialize_value]
#axiomsClean materialize_code
#axiomsClean materialize_value
#axiomsClean materialize_cut
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Native
