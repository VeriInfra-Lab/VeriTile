import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Mathematical

namespace VeriTile.Bench.Repairs.ContextAttnMistral.WholeReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Checked Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Full Real AST computation composed with direct dynamic physical stores.
The finite -1e9 terms remain in the denominator, including padded key lanes. -/
theorem correct (api : API ℝ Store Byte) (adapter : Mathematical.Adapter c tokens)
    (accepted : check c tokens request buffers=Bool.true)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store) :
    (∀ q,api.read (execute api 0 c tokens buffers (Mathematical.source adapter) order m) (address buffers q)=
      Mathematical.target c (inputs api 0 c tokens buffers m) q) ∧
    (∀ i,Outside buffers order i →
      api.byte (execute api 0 c tokens buffers (Mathematical.source adapter) order m) i=api.byte m i) := by
  have a := check_iff.mp accepted
  constructor
  · intro q
    rw [readback api 0 a _ order m q (complete q),Mathematical.source_target a.geometry]
  · exact byte_frame api 0 a _ order m

/-- The public wrapper result includes the successful empty branch, the None
return and exact supplied output descriptor, together with every physical byte. -/
theorem checked_call (api : API ℝ Store Byte) (adapter : Mathematical.Adapter c tokens)
    (launch : Bool) (accepted : validate c tokens request buffers=.ok launch)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store) :
    ∃ after, dispatch api 0 c tokens request buffers (Mathematical.source adapter) order m=
      .ok (none,request.tensor .output,after) ∧
      ((launch=Bool.false ∧ after=m) ∨
       (launch=Bool.true ∧
        (∀ q,api.read after (address buffers q)=Mathematical.target c (inputs api 0 c tokens buffers m) q) ∧
        (∀ i,Outside buffers order i → api.byte after i=api.byte m i))) := by
  cases launch
  · exact ⟨m,dispatch_empty _ _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have a := (validate_launch.mp accepted).2
    have result := correct api adapter (check_iff.mpr a) order complete m
    exact ⟨_,dispatch_launch _ _ _ _ _ _ _ _ _ accepted,Or.inr ⟨rfl,result⟩⟩

#axiomsClean correct
#axiomsClean checked_call
#auditStmt checked_call
#specNonCircular Mathematical.target avoiding [stepStmts,execute,dispatch,Mathematical.source]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.WholeReal
