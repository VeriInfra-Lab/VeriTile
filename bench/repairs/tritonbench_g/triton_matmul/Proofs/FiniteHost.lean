import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteGrid
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteHost
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter
noncomputable section
set_option maxHeartbeats 6000000

def call {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (allocate : Allocator α)
    (a b : Tensor) (before : State α) : Option (Tensor × State α) :=
  if checkTensorInputs a.metadata b.metadata=.ok () then
    (allocate a b before).bind fun (c,prepared) =>
      if Contract.Empty a.metadata b.metadata then some (c,prepared)
      else (FiniteGrid.grid M D (request a b) (List.range (programs (request a b))) prepared).map (fun final => (c,final))
  else none

theorem correct {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (Q : Numerics α)
    (allocate : Allocator α) (read : Read) (decode : TensorBinding.Word → α) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata)
    (laws : PrimitiveLaws M D Q (storage a.metadata)) (before : State α)
    (allocates : AllocationLaw allocate a b before) (bound : Inputs read decode a b before) :
    ∃ c prepared final,allocate a b before=some (c,prepared) ∧
      call M D allocate a b before=some (c,final) ∧
      CheckedHost.Post a b c before final (fun i => logicalReference Q (request a b)
        (CheckedHost.left read decode a b la) (CheckedHost.right read decode a b lb) i.1 i.2) := by
  obtain ⟨c,prepared,typed,allocated,called,post,source⟩ :=
    CheckedHost.correct M D Q allocate read decode a b la lb h laws before allocates bound
  have checked := (checkTensorInputs_iff a.metadata b.metadata).mpr h
  by_cases empty : Contract.Empty a.metadata b.metadata
  · have same : prepared=typed := by
      simp only [CheckedHost.call,if_pos checked,allocated,Option.bind_some,if_pos empty] at called
      exact (Prod.mk.inj (Option.some.inj called)).2
    subst typed
    exact ⟨c,prepared,prepared,allocated,by simp [call,checked,allocated,empty],post⟩
  · obtain ⟨tensor,he,safety,correspondence⟩ := source empty
    have launch : programs (request a b)<2^31 :=
      (execution_nonempty a.metadata b.metadata h empty).1.2.2.2
    obtain ⟨actual,other,ha,ho,hm⟩ := FiniteGrid.source M D (request a b) prepared tensor launch
      (List.range (programs (request a b))) (fun _ hp => List.mem_range.mp hp) prepared prepared rfl
    have typed_eq : other=typed := by
      have st : (request a b).storage=(request a b).storageB := tensor.storage
      simp only [host,if_neg (not_not.mpr tensor.dimensions),if_neg (not_not.mpr st),ho] at he
      exact (Prod.mk.inj (Except.ok.inj he)).2
    subst other
    refine ⟨c,prepared,actual,allocated,?_,⟨post.metadata,?_,?_⟩⟩
    · simp [call,checked,allocated,empty,ha]
    · intro i; rw [hm]; exact post.values i
    · intro r o outside; rw [hm]; exact post.frame r o outside

#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.FiniteHost
