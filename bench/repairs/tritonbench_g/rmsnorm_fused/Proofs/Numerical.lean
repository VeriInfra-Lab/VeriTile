import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Physical

namespace VeriTile.Bench.Repairs.RmsnormFused.Numerical
open VeriTile Triton FP.Structural Native
noncomputable section
set_option maxHeartbeats 6000000

/-- The seven scalar operation families actually reached by this source. A
literal conversion covers 0, 1, the bounded N conversion and runtime epsilon.
Loads are raw-word wires; tree shape and FMA selection are structural choices. -/
structure ScalarOps where
  literal : ℝ → Word
  add : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  sqrt : Word → Word
  castFP32 : Word → Word
  fmaRN : Word → Word → Word → Word

def lower (p : ScalarOps) : Instructions where
  literal := fun _ _ x => p.literal x
  binary := fun _ _ op x y => match op with
    | .add => p.add x y
    | .mul => p.mul x y
    | .div => p.div x y
    | _ => 0
  unary := fun _ op x => match op with
    | .sqrt => p.sqrt x
    | _ => 0
  cast := fun _ _ _ x => p.castFP32 x
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fmaRN := p.fmaRN

/-- These are scalar equalities, universally quantified over raw operands.
There is no reduction, output, kernel, allocation or native-to-Real callback. -/
structure ScalarLaws (actual reference : ScalarOps) : Prop where
  literal : ∀ x,actual.literal x=reference.literal x
  add : ∀ x y,actual.add x y=reference.add x y
  mul : ∀ x y,actual.mul x y=reference.mul x y
  div : ∀ x y,actual.div x y=reference.div x y
  sqrt : ∀ x,actual.sqrt x=reference.sqrt x
  castFP32 : ∀ x,actual.castFP32 x=reference.castFP32 x
  fmaRN : ∀ x y z,actual.fmaRN x y z=reference.fmaRN x y z

theorem ScalarLaws.operations_eq (a b : ScalarOps) (h : ScalarLaws a b) : a=b := by
  cases a
  cases b
  simp only [ScalarOps.mk.injEq]
  exact ⟨funext h.literal,funext (fun x => funext (h.add x)),
    funext (fun x => funext (h.mul x)),funext (fun x => funext (h.div x)),
    funext h.sqrt,funext h.castFP32,funext (fun x => funext (fun y => funext (h.fmaRN x y)))⟩

/-- Reachability excludes every generic unused Algebra port, including max,
exp, fromNat/fromInt, bit construction, and all nonidentity casts. -/
def supported : Term → Prop
  | .leaf _ _ _ => True
  | .literal p d _ => p=some .fp32 ∧ d=.real
  | .binary p d op a b => p=some .fp32 ∧ d=.real ∧
      (op=.add ∨ op=.mul ∨ op=.div) ∧ supported a ∧ supported b
  | .unary p op a => p=some .fp32 ∧ op=.sqrt ∧ supported a
  | .cast p src dst a => p=some .fp32 ∧ src=.real ∧ dst=.real ∧ supported a
  | .load a => supported a
  | _ => False

theorem tree_supported (t : Native.Tree n) (xs : Fin n → Term) (hx : ∀ i,supported (xs i)) :
    supported (t.expand (some .fp32) .add xs (.literal (some .fp32) .real 0)) := by
  induction t with
  | lane i => exact hx i
  | neutral => exact ⟨rfl,rfl⟩
  | join a b ha hb => exact ⟨rfl,rfl,Or.inl rfl,ha,hb⟩

theorem reduction_supported (R : Reductions) {s : TileShape} (ax : Fin s.length)
    (keep : Bool) (xs : Values Term .real s)
    (out : TileIndex (TileShape.reduceShape s ax keep)) (hx : ∀ i,supported (xs i)) :
    supported (reduction R Bool.false (some .fp32) ax keep xs out) := by
  apply tree_supported
  intro i
  cases keep <;> exact hx _

theorem value_supported (R : Reductions) (memory : RegionName → Nat → Word)
    (X W : RegionName) (row N B : Nat) (eps : ℝ) (k : Nat) (hk : k<N) :
    supported (Row.value (algebra R) (entry memory) X W row N B eps k) := by
  have hx : ∀ j,supported (Row.xload (algebra R) (entry memory) X row N j) := by
    intro j
    by_cases hj : j<N <;>
      simp [Row.xload,Row.cast,Row.read,Row.lit,algebra,entry,Cell.read_same,supported,hj]
  have sq : ∀ i,supported (Row.square (algebra R) (entry memory) X row N B i) := by
    intro i
    by_cases hi : i.1.val<N <;>
      simp [Row.square,Row.add,Row.mul,Row.lit,algebra,supported,hi]
    exact hx _
  have red := reduction_supported R (s:=[B]) ⟨0,by simp⟩ Bool.false
    (Row.square (algebra R) (entry memory) X row N B) PUnit.unit sq
  simp [Row.value,Row.mul,Row.rstd,Row.div,Row.add,Row.lit,Row.variance,Row.wload,
    Row.read,algebra,entry,Cell.read_same,supported,hk]
  exact ⟨hx k,red⟩

/-- Full returned bytes follow from precisely these seven scalar families.
The actual valid reduction tree and fusion/operand schedule are kept unchanged. -/
theorem whole_equiv (actual reference : ScalarOps) (laws : ScalarLaws actual reference)
    (R : Reductions) (S : Schedule) (q : Host.Request) (memory : Physical.Bytes)
    (call : Host.Preparation (State Term) (Cell Term) State.mem q)
    (hbefore : call.before=entry (Physical.load memory)) (schedule : List Nat)
    (all : schedule.Perm q.geometry.grid) :
    ∃ result,Host.invoke (algebra R) q call schedule=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=q.shape ∧
      result.tensor.dtype="float32" ∧ result.tensor.device=call.input.device ∧
      result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.rows=0 then none else some call.input.device) ∧
      (∀ row k,row<q.rows → k<q.columns → ∀ i : Fin 4,
        Physical.project (lower actual) S result.state result.tensor.region (4*(row*q.columns+k)+i.val)=
          Physical.encode (value (lower reference) R S (Physical.load memory)
            call.input.region call.weight.region q row k) i) ∧
      (∀ r b,r≠result.tensor.region ∨ 4*q.size≤b →
        Physical.project (lower actual) S result.state r b=memory r b) := by
  cases laws.operations_eq actual reference
  exact Physical.whole_bytes (lower actual) R S q memory call hbefore schedule all

#axiomsClean ScalarLaws.operations_eq
#axiomsClean value_supported
#axiomsClean whole_equiv
#specNonCircular ScalarLaws avoiding [Host.invoke,Row.source,FP.Structural.exec,Native.value]
end
end VeriTile.Bench.Repairs.RmsnormFused.Numerical
