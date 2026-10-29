import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RealExecution

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Real
open VeriTile Triton
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Checked ideal-Real call through complete typed source execution, with
fresh zero output (including inactive tails), a mathematical sparse-attention
target, and the complement byte frame. Integer interpretation is a local
input-word ABI operation; it has no source/result premise. -/
theorem checked_call (api : API ℝ Store Byte) (c : Config) (layout : Reify.Layout c)
    (integer : ℝ → Int) (request : Request) (m : Store) (launch : Bool)
    (accepted : validate c request (RealExecution.observed c integer (Raw.inputs api 0 c request m))=.ok launch)
    (order : List (Raw.Coord c)) (complete : ∀ q,q∈order) :
    ∃ after,Raw.dispatch api 0 c request (RealExecution.observed c integer (Raw.inputs api 0 c request m))
      (RealExecution.source c layout integer) order m=.ok (request.tensor .output,after) ∧
      ((launch=Bool.false ∧ after=m) ∨ (launch=Bool.true ∧
        (∀ q : Raw.Coord c,api.read after (Raw.address request q)=
          RealExecution.target c layout integer (Raw.inputs api 0 c request m) q) ∧
        (∀ i,Raw.Outside request order i → api.byte after i=api.byte m i) ∧
        Raw.inputs api 0 c request after=Raw.inputs api 0 c request m)) := by
  cases launch
  · exact ⟨m,Raw.dispatch_empty _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have live:=(validate_launch.mp accepted).2
    refine ⟨_,Raw.dispatch_launch _ _ _ _ _ _ _ _ accepted,Or.inr ⟨rfl,?_,
      Raw.byte_frame api 0 live _ order m,Raw.inputs_execute api 0 live _ order m⟩⟩
    intro q
    rw [Raw.readback api 0 live _ order m q (complete q),RealExecution.source_target c layout integer request _ live]

#axiomsClean checked_call
#auditStmt checked_call
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Real
