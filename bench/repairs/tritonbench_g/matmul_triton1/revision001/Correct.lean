import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.PublicContract
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open VeriTile.Triton FP.Structural
noncomputable section



/-- Complete mathematical matrix multiplication, including allocation of the
actual returned tensor, zero-sized branches, row-major grid and full frame.
The only allocation premise is the ordinary local torch.empty contract. -/
specification matmul_triton1_correct (rest : Algebra ℝ) (allocate : Host.Allocator ℝ)
    (allocation : Host.AllocationLaw allocate) (x : Host.Arguments)
    (h : Host.Valid x) (initial : State ℝ)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest Core.realPrimitives allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.realMatmul (Host.parameters x c) initial i.1.val i.2.val) := by
  obtain ⟨c,final,he,hpost⟩ := Host.correct rest Core.realPrimitives allocate allocation x h initial
  refine ⟨c,final,he,?_⟩
  simpa only [Host.Post,OutputPost,Core.logical_realMatmul (Host.parameters x c) initial _ _ h.blockInner]
    using hpost

#axiomsClean matmul_triton1_correct
/-- Complete checked finite64 wrapper, with the independent mathematical sum
of the original input Tensor logical values and exact returned allocation. -/
specification matmul_triton1_checked_real (rest : Algebra ℝ) (decode : TensorBinding.Word → ℝ)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (default : Bool) (h : InputContract default a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State ℝ)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read decode initial)
    (allocate : Host.Allocator ℝ)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb default) initial)
    (launch : CheckedHost.Launchable allocate (InputAdapter.arguments a b ra rb default) initial) :
    ∃ c final, CheckedHost.call Core.realPrimitives allocate a b ra rb default initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb default) initial final c
        (fun i => LogicalInputs.realExpected a b la lb read decode i.1.val i.2.val) := by
  obtain ⟨c,final,he,post⟩ := CheckedHost.correct rest Core.realPrimitives allocate
    a b la lb ra rb default h initial allocation launch
  refine ⟨c,final,he,?_⟩
  have refs : ∀ i : Host.Index (InputAdapter.arguments a b ra rb default),
      Core.logical Core.realPrimitives (Host.parameters (InputAdapter.arguments a b ra rb default) c)
        initial i.1.val i.2.val = LogicalInputs.realExpected a b la lb read decode i.1.val i.2.val := by
    intro i
    rw [Core.logical_realMatmul _ _ _ _ (InputAdapter.valid a b ra rb default h).blockInner]
    exact LogicalInputs.real_reference a b la lb ra rb default h read decode initial bound c i
  simpa only [Host.Post,OutputPost,refs] using post

#axiomsClean matmul_triton1_checked_real

end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001
