import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Checked
import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Mathematical

namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Checked full source execution, all gathered mathematical scores, caller
output identity, None return, all source integer bounds and finite cell frame.
Only ordinary current-storage binding and successful runtime remain external. -/
theorem correct (c : Call) (m : Metadata c) (r : Request c)
    (accepted : check c m r=Bool.true) (x : Buffers ℝ c) (s : State ℝ)
    (flat : Fin (2^62) → Cell ℝ)
    (binding : FlatBinding c (checkedPlacement c m r ((check_iff c m r).mp accepted)) m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    let a := (check_iff c m r).mp accepted
    Formed c m ∧ ∃ result,
      invoke (fun _ => realAlgebra) c (idealScale c.dim) (checkedABI c m r a) work (initial c m x s)=some result ∧
      result.returned=none ∧ result.output=r.tensor .output ∧
      Result c m (realFormula c (idealScale c.dim) m x) (initial c m x s) result.state ∧
      FlatResult c (checkedPlacement c m r a) m (realFormula c (idealScale c.dim) m x) flat
        (flatten (checkedPlacement c m r a) flat result.state) := by
  let a := (check_iff c m r).mp accepted
  exact mathematical_correct c m x s (checkedABI c m r a) a.dense a.guard
    (checked_data c m a.metadata) (checkedPlacement c m r a) flat binding work legal

/-- Both successful Python validation branches reach this public endpoint.
No dense stride or typed physical binding is requested of an empty call. -/
theorem checked_call (c : Call) (m : Metadata c) (r : Request c) (launch : Bool)
    (accepted : validate c m r=.ok launch) (x : Buffers ℝ c) (s : State ℝ)
    (flat : Fin (2^62) → Cell ℝ)
    (binding : ∀ a : Admission c m r, FlatBinding c (checkedPlacement c m r a) m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    ∃ result, dispatch (fun _=>realAlgebra) c m r (idealScale c.dim) work (initial c m x s)=.ok (some result) ∧
      result.returned=none ∧ result.output=r.tensor .output ∧
      ((launch=Bool.false ∧ result.state=initial c m x s) ∨
       (∃ a : Admission c m r, launch=Bool.true ∧ Formed c m ∧
        Result c m (realFormula c (idealScale c.dim) m x) (initial c m x s) result.state ∧
        FlatResult c (checkedPlacement c m r a) m (realFormula c (idealScale c.dim) m x) flat
          (flatten (checkedPlacement c m r a) flat result.state))) := by
  cases launch
  · exact ⟨⟨initial c m x s,r.tensor .output,none⟩,
      dispatch_empty _ _ _ _ _ _ _ accepted,rfl,rfl,Or.inl ⟨rfl,rfl⟩⟩
  · have a := ((validate_launch c m r).mp accepted).2
    have hc := (check_iff c m r).mpr a
    obtain ⟨formed,result,run,ret,out,res,frame⟩ := correct c m r hc x s flat (binding _) work legal
    refine ⟨result,?_,ret,out,Or.inr ⟨a,rfl,formed,res,frame⟩⟩
    rw [dispatch_launch _ _ _ _ _ _ _ accepted]
    exact congrArg Except.ok run

#axiomsClean checked_call
#axiomsClean correct
#auditStmt correct
#specNonCircular realFormula avoiding [FP.Structural.exec,invoke,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
