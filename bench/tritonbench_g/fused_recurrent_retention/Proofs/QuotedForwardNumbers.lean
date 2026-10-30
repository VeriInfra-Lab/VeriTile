import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedValues

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2500000

@[simp] theorem literal_zero (a : Arithmetic α) : Staged.literal a 0=Staged.atom a.zero := by
  norm_num [Staged.literal]
@[simp] theorem atom_value (x : α) : (Staged.atom x).value=x := rfl
@[simp] theorem tagged_value (k : Staged.Kind) (x : α) : (Staged.tagged k x).value=x := rfl
@[simp] theorem product_value (a : Arithmetic α) (k : Staged.Kind) (x y : α) :
    (Staged.product a k x y).value=a.mul x y := rfl
@[simp] theorem node_ite_value (c : Prop) [Decidable c] (x y : Staged.Node α) :
    (if c then x else y).value=(if c then x.value else y.value) := by split <;> rfl

/-- The arithmetic expression emitted by the source body, independently of
register updates, scatter order and the target dot recurrence. -/
def forwardUpdateNode (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) : Staged.Node α :=
  Staged.add a
    (Staged.multiply a (Staged.tagged .beta (decay a (pid.head%s.H)))
      (carryNode a s x seed pid t key value))
    (Staged.multiply a (Staged.atom (keyLoad a s x.k pid.head t (pid.key s key)))
      (Staged.atom (valueLoad a s x.v pid.head t (pid.value s value))))

@[simp] theorem forwardUpdateNode_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) :
    forwardUpdateNode a s x seed pid t key value = carryNode a s x seed pid (t+1) key value := by
  simp [forwardUpdateNode,carryNode_succ]

def forwardProductNode (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) : Staged.Node α :=
  Staged.multiply a (forwardUpdateNode a s x seed pid t key value)
    (Staged.multiply a (Staged.atom (keyLoad a s x.q pid.head t (pid.key s key)))
      (Staged.tagged .scale (a.scale s.DK)))

@[simp] theorem forwardProductNode_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) :
    forwardProductNode a s x seed pid t key value =
      Staged.product a .outputDot (programForwardCarry a s x seed pid (t+1) key value)
        (a.mul (keyLoad a s x.q pid.head t (pid.key s key)) (a.scale s.DK)) := by
  simp only [forwardProductNode,forwardUpdateNode_eq,Staged.scaled_query]
  simp [carryNode,Staged.forward_product]

theorem forwardSum_eq (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t : Nat) (plans : Plans) (hBK : 0<s.BK) (covered : plans.output.Covers s.BK) :
    Staged.sumValues a plans (shape := [s.BV,s.BK]) ⟨1,by simp⟩ Bool.false
      (fun i => forwardProductNode a s x seed pid t i.2.1.val i.1.val) =
      fun i => Staged.atom (programO a s plans x seed pid t i.1.val) := by
  simp only [forwardProductNode_eq]
  exact Staged.output_reduction a plans s.BK s.BV hBK covered
    (fun v k => programForwardCarry a s x seed pid (t+1) k v)
    (fun k => a.mul (keyLoad a s x.q pid.head t (pid.key s k)) (a.scale s.DK))

#axiomsClean forwardUpdateNode_eq
#axiomsClean forwardProductNode_eq
#axiomsClean forwardSum_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
