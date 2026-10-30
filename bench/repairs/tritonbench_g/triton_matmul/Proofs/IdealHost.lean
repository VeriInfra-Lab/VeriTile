import bench.repairs.tritonbench_g.triton_matmul.Proofs.IdealTensor
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.IdealHost
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter IdealTensor
noncomputable section
set_option maxHeartbeats 6000000
theorem typed_correct (allocate : Allocator ℝ) (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata)
    (before : State ℝ)
    (allocates : AllocationLaw allocate a b before) (bound : Inputs read a b before) :
    ∃ c prepared final,allocate a b before=some (c,prepared) ∧
      CheckedHost.call ideal idealDot allocate a b before=some (c,final) ∧
      CheckedHost.Post a b c before final (fun i => logicalReference idealNumerics (request a b)
        (left read a b) (right read a b) i.1 i.2) ∧
      (¬Contract.Empty a.metadata b.metadata → ∃ tensor : TensorAPI (request a b) prepared,
        host ideal idealDot (request a b) prepared=.ok (outputC,final) ∧
        AddressSafety (request a b) prepared tensor ∧ Index.Correspondence (request a b) prepared tensor) := by
  obtain ⟨c,prepared,allocated,allocation⟩ := allocates
  have checked := (checkTensorInputs_iff a.metadata b.metadata).mpr h
  by_cases empty : Contract.Empty a.metadata b.metadata
  · refine ⟨c,prepared,prepared,allocated,?_,⟨allocation.metadata,?_,allocation.frame⟩,fun ne => False.elim (ne empty)⟩
    · simp [CheckedHost.call,checked,allocated,empty]
    · intro i
      have ir := i.1.isLt; have ic := i.2.isLt
      rcases empty with e | e <;> omega
  · let tensor := tensorAPI read a b c h empty before prepared allocation bound
    have bounds := execution_nonempty a.metadata b.metadata h empty
    have launch : programs (request a b)<2^31 := by
      simpa only [programs,tileRows,tileColumns,request] using bounds.1.2.2.2
    have preparedInputs := allocated_inputs read a b c before prepared allocation bound
    have logical := source_inputs read a b la lb h prepared preparedInputs
    obtain ⟨final,he,values,frame⟩ := host_correct ideal idealDot idealNumerics (request a b) (ideal_primitives _) prepared
      tensor.freshA tensor.freshB tensor.dimensions tensor.storage
    refine ⟨c,prepared,final,allocated,?_,⟨allocation.metadata,?_,?_⟩,?_⟩
    · simp [CheckedHost.call,checked,allocated,empty,he,Except.toOption]
    · intro i
      have hv := values i.1.val i.1.isLt i.2.val i.2.isLt
      rw [reference_logical idealNumerics (request a b) prepared _ _ logical.1 logical.2 i.1 i.2] at hv
      have hn := i.2.isLt
      simpa only [request,outputOffset,address,max_eq_left (show 1≤columns b.metadata by omega),Nat.mul_comm] using hv
    · intro r o outside
      apply (frame r o ?_).trans (allocation.frame r o outside)
      rintro ⟨hr,ho⟩
      rcases outside with ne | large
      · exact ne hr
      · have size := output_numel a b c allocation.metadata
        obtain ⟨row,hr,col,hcol,rfl⟩ := ho
        have ho := linear_bound row col (rows a.metadata) (columns b.metadata) hr hcol
        have pos : 1≤columns b.metadata := by change col<columns b.metadata at hcol; omega
        simp only [outputOffset,request,max_eq_left pos] at large
        rw [size] at large
        nlinarith
    · exact fun _ => ⟨tensor,he,address_safety _ _ tensor,Index.correspondence _ _ tensor launch⟩

theorem correct (allocate : Allocator ℝ) (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata)
    (before : State ℝ)
    (allocates : AllocationLaw allocate a b before) (bound : Inputs read a b before) :
    ∃ c prepared final,allocate a b before=some (c,prepared) ∧
      FiniteHost.call ideal idealDot allocate a b before=some (c,final) ∧
      CheckedHost.Post a b c before final (fun i => logicalReference idealNumerics (request a b)
        (left read a b) (right read a b) i.1 i.2) := by
  obtain ⟨c,prepared,typed,allocated,called,post,source⟩ :=
    typed_correct allocate read a b la lb h before allocates bound
  have checked := (checkTensorInputs_iff a.metadata b.metadata).mpr h
  by_cases empty : Contract.Empty a.metadata b.metadata
  · have same : prepared=typed := by
      simp only [CheckedHost.call,if_pos checked,allocated,Option.bind_some,if_pos empty] at called
      exact (Prod.mk.inj (Option.some.inj called)).2
    subst typed
    exact ⟨c,prepared,prepared,allocated,by simp [FiniteHost.call,checked,allocated,empty],post⟩
  · obtain ⟨tensor,he,safety,correspondence⟩ := source empty
    have launch : programs (request a b)<2^31 :=
      (execution_nonempty a.metadata b.metadata h empty).1.2.2.2
    obtain ⟨actual,other,ha,ho,hm⟩ := FiniteGrid.source ideal idealDot (request a b) prepared tensor launch
      (List.range (programs (request a b))) (fun _ hp => List.mem_range.mp hp) prepared prepared rfl
    have typed_eq : other=typed := by
      have st : (request a b).storage=(request a b).storageB := tensor.storage
      simp only [host,if_neg (not_not.mpr tensor.dimensions),if_neg (not_not.mpr st),ho] at he
      exact (Prod.mk.inj (Except.ok.inj he)).2
    subst other
    refine ⟨c,prepared,actual,allocated,?_,⟨post.metadata,?_,?_⟩⟩
    · simp [FiniteHost.call,checked,allocated,empty,ha]
    · intro i; rw [hm]; exact post.values i
    · intro r o outside; rw [hm]; exact post.frame r o outside

#axiomsClean typed_correct
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.IdealHost
