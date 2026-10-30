import bench.tritonbench_g.fused_recurrent_retention.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 1800000

/-- Independent dense prefix/suffix mathematics, restricted to each actual
partial-output tile; no host reduction is claimed by a kernel theorem. -/
def mathCell (s : Shape) (x : Inputs ℝ) (seed final : Bool) (kind : PartialKind) (i : Nat) : ℝ :=
  let w := kind.width s
  let h := flatHead s w i
  let t := flatTime s w i
  let part := flatSplit s w i
  let c := flatChannel w i
  match kind with
  | .output => ∑ j∈Finset.range s.BK,if part*s.BK+j<s.DK then
      mathState s x seed h (part*s.BK+j) c (t+1)*(x.q h t (part*s.BK+j)*realArithmetic.scale s.DK) else 0
  | .dq => (∑ j∈Finset.range s.BV,if part*s.BV+j<s.DV then
      mathState s x seed h c (part*s.BV+j) (t+1)*x.do_ h t (part*s.BV+j) else 0)*realArithmetic.scale s.DK
  | .dk => ∑ j∈Finset.range s.BV,if part*s.BV+j<s.DV then
      mathAdjoint s x final h t c (part*s.BV+j)*x.v h t (part*s.BV+j) else 0
  | .dv => ∑ j∈Finset.range s.BK,if part*s.BK+j<s.DK then
      mathAdjoint s x final h t (part*s.BK+j) c*x.k h t (part*s.BK+j) else 0
  | .finalState => mathState s x seed (i/s.DV/s.DK) (i/s.DV%s.DK) (i%s.DV) s.T

def mathExpected (s : Shape) (seed final : Bool)
    (xs : ChannelValues (Staged.Node ℝ) (inputPorts s)) : ChannelValues ℝ (outputPorts s) :=
  fun i => match i with
  | ⟨0,_⟩ => fun j => mathCell s (logicalInputs realArithmetic s xs) seed final .output j.1.val
  | ⟨1,_⟩ => fun j => mathCell s (logicalInputs realArithmetic s xs) seed final .dq j.1.val
  | ⟨2,_⟩ => fun j => mathCell s (logicalInputs realArithmetic s xs) seed final .dk j.1.val
  | ⟨3,_⟩ => fun j => mathCell s (logicalInputs realArithmetic s xs) seed final .dv j.1.val
  | ⟨4,_⟩ => fun j => mathCell s (logicalInputs realArithmetic s xs) seed final .finalState j.1.val

theorem expected_real_eq (s : Shape) (p : PlanBinding s) (seed final : Bool) :
    expected realArithmetic s p seed final=mathExpected s seed final := by
  funext xs o j
  fin_cases o
  · obtain ⟨_,_,_,hv,_⟩ := unflatten_partial s s.NK s.DV j.1.val j.1.isLt
    exact partialO_real s p.plans p.valid _ seed _ _ _ _ hv
  · obtain ⟨_,_,_,hk,_⟩ := unflatten_partial s s.NV s.DK j.1.val j.1.isLt
    exact partialQ_real s p.plans p.valid _ seed _ _ _ _ hk
  · obtain ⟨_,_,ht,hk,_⟩ := unflatten_partial s s.NV s.DK j.1.val j.1.isLt
    exact partialK_real s p.plans p.valid _ final _ _ _ _ hk ht
  · obtain ⟨_,_,ht,hv,_⟩ := unflatten_partial s s.NK s.DV j.1.val j.1.isLt
    exact partialV_real s p.plans p.valid _ final _ _ _ _ hv ht
  · obtain ⟨_,hk,hv,_⟩ := unflatten_state s j.1.val j.1.isLt
    exact state_real s _ seed _ _ _ _ hk hv

theorem real_complete (mode : Mode) (s : Shape) (shape : s.Admitted)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (io mode realArithmetic s p incoming finalGradient seed final).Observes
      (fun _ _ n => n.value) (mathExpected s seed final) := by
  rw [←expected_real_eq s p seed final]
  exact complete mode realArithmetic s shape p incoming finalGradient seed final

#axiomsClean real_complete
#specNonCircular mathCell avoiding [partialValue,state,reverseCarry,adjoint,sourceCell,Quoted.gridExec]
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
