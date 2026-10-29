import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Whole

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Native
noncomputable section
set_option maxHeartbeats 8000000

/-- Source expressions retain each FP32 operation. Leaf addresses identify loads;
no output value or interpreter result is present in the compiler policy. -/
inductive Shape where
  | zero
  | leaf (region : RegionName) (address : Nat)
  | add (lhs rhs : Shape)
  | mul (lhs rhs : Shape)
  deriving Inhabited

inductive Term where
  | zero
  | leaf (region : RegionName) (address : Nat) (word : BitVec 32)
  | add (lhs rhs : Term)
  | mul (lhs rhs : Term)
  deriving Inhabited

def shape : Term → Shape
  | .zero => .zero
  | .leaf r a _ => .leaf r a
  | .add a b => .add (shape a) (shape b)
  | .mul a b => .mul (shape a) (shape b)

def sourceOps : Ops Term := ⟨.zero,.add,.mul⟩

/-- Three individual native FP32 instructions, whose IEEE meanings are local
numerical obligations. No recurrence or array result is an instruction port. -/
structure Instructions where
  addRN : BitVec 32 → BitVec 32 → BitVec 32
  mulRN : BitVec 32 → BitVec 32 → BitVec 32
  fmaRN : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32

/-- Ordinary compiler choices. The policy reads source topology only, never raw
input values. It can vary by specialization and expression site. In particular
it does not assume that every reverse-loop boundary contracts to FMA. -/
structure Schedule where
  swapAdd : Shape → Shape → Bool
  swapMul : Shape → Shape → Bool
  contract : Shape → Shape → Shape → Bool
  swapFma : Shape → Shape → Shape → Bool

def addValue (p : Instructions) (S : Schedule) (a b : Term) (x y : BitVec 32) :=
  if S.swapAdd (shape a) (shape b) then p.addRN y x else p.addRN x y

def eval (p : Instructions) (S : Schedule) : Term → BitVec 32
  | .zero => 0
  | .leaf _ _ w => w
  | .mul a b => if S.swapMul (shape a) (shape b) then p.mulRN (eval p S b) (eval p S a)
      else p.mulRN (eval p S a) (eval p S b)
  | .add a c => match a with
    | .mul x y => if S.contract (shape x) (shape y) (shape c) then
        if S.swapFma (shape x) (shape y) (shape c)
        then p.fmaRN (eval p S y) (eval p S x) (eval p S c)
        else p.fmaRN (eval p S x) (eval p S y) (eval p S c)
      else addValue p S (.mul x y) c (eval p S (.mul x y)) (eval p S c)
    | a => addValue p S a c (eval p S a) (eval p S c)

def observe (p : Instructions) (S : Schedule) (s : State Term) (r : RegionName) (a : Nat) : BitVec 32 :=
  eval p S ((s.mem r a).read .real)

theorem eval_fused (p : Instructions) (S : Schedule) (x y z : Term)
    (h : S.contract (shape x) (shape y) (shape z)=Bool.true) :
    eval p S (.add (.mul x y) z)=
      if S.swapFma (shape x) (shape y) (shape z)
      then p.fmaRN (eval p S y) (eval p S x) (eval p S z)
      else p.fmaRN (eval p S x) (eval p S y) (eval p S z) := by simp [eval,h]

theorem eval_split (p : Instructions) (S : Schedule) (x y z : Term)
    (h : S.contract (shape x) (shape y) (shape z)=Bool.false) :
    eval p S (.add (.mul x y) z)=addValue p S (.mul x y) z
      (if S.swapMul (shape x) (shape y) then p.mulRN (eval p S y) (eval p S x)
        else p.mulRN (eval p S x) (eval p S y)) (eval p S z) := by simp [eval,h]

/-- Whole source graph result observed under an arbitrary local native schedule.
Binding a concrete binary to this schedule is an ordinary compiler contract;
it is not assumed as a whole-output equation in this theorem. -/
theorem whole_native (p : Instructions) (schedule : Schedule) (input : Input Term)
    (BH T D BDF BDB : Nat) (hF : 0 < BDF) (hB : 0 < BDB) (U S V : Bool)
    (s : State Term) (h : Execution.Inputs s input T D) :
    ∃ f b,
      Execution.gridRun sourceOps (forward T D BDF U S).surfaceBody (Execution.programs BH D BDF) s=some f ∧
      Execution.gridRun sourceOps (backward T D BDB U V).surfaceBody (Execution.programs BH D BDB) f=some b ∧
      (∀ bh, bh < BH → ∀ t, t < T → ∀ d, d < D →
        observe p schedule b "o" (rowAddr bh T D t d)=eval p schedule (state sourceOps U input bh d (t+1)) ∧
        observe p schedule b "dx" (rowAddr bh T D t d)=eval p schedule (dx sourceOps V input T bh d t) ∧
        observe p schedule b "dg" (rowAddr bh T D t d)=eval p schedule (dg sourceOps U V input T bh d t)) ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        observe p schedule b "ht" (stateAddr bh D d)=eval p schedule (state sourceOps U input bh d T)) ∧
      (∀ r a, ¬ Execution.ForwardRegion BH T D S r a → ¬ Execution.BackwardRegion BH T D r a →
        b.mem r a=s.mem r a) := by
  obtain ⟨f,b,hf,hb,hO,hG,hH,hframe⟩ := Execution.whole_result sourceOps input BH T D BDF BDB hF hB U S V s h
  refine ⟨f,b,hf,hb,?_,?_,hframe⟩
  · intro bh hbh t ht d hd
    obtain ⟨hx,hg⟩ := hG bh hbh t ht d hd
    unfold observe
    rw [hO bh hbh t ht d hd,hx,hg]
    exact ⟨rfl,rfl,rfl⟩
  · intro hS bh hbh d hd
    unfold observe
    rw [hH hS bh hbh d hd]
    rfl

#specNonCircular eval avoiding [FP.Structural.exec,Execution.gridRun]
#axiomsClean whole_native
end
end Native
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
