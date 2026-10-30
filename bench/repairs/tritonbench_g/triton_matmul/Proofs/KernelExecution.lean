import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelSource
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter KernelContract
noncomputable section
set_option maxHeartbeats 1000000

theorem empty_programs (a b : Tensor) (empty : Contract.Empty a.metadata b.metadata) :
    programs (request a b)=0 := by
  rcases empty with e | e <;> simp [programs,tileRows,tileColumns,request,e]

/-- Full shared DSL execution and finite signed-64 source execution, with equal
whole memories. Success is derived; broad primitive laws are an internal lemma
interface and are discharged by the public Real/native theorems. -/
theorem executes {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (Q : Numerics α)
    (read : Storage → Nat → α) (a b c : Tensor) (h : InputContract a.metadata b.metadata)
    (initial : State α) (output : Output a b c) (bound : KernelContract.Inputs read a b initial)
    (laws : PrimitiveLaws M D Q (storage a.metadata)) :
    ∃ actual typed,
      FiniteGrid.grid M D (request a b) (List.range (programs (request a b))) initial=some actual ∧
      runPrograms M D (request a b) (List.range (programs (request a b))) initial=some typed ∧
      actual.mem=typed.mem ∧ Result Q (request a b) initial actual := by
  obtain ⟨typed,ht,result⟩ := grid_correct M D Q (request a b) laws initial
    (by simp [request,outputC,TensorBinding.inputA])
    (by simp [request,outputC,TensorBinding.inputB])
  by_cases empty : Contract.Empty a.metadata b.metadata
  · have np := empty_programs a b empty
    have eq : initial=typed := by simpa only [np,List.range_zero,runPrograms,Option.some.injEq] using ht
    subst typed
    exact ⟨initial,initial,by simp [np,FiniteGrid.grid],ht,rfl,result⟩
  · let tensor := tensorAPI read a b c h empty initial output bound
    have launch : programs (request a b)<2^31 := (execution_nonempty _ _ h empty).1.2.2.2
    obtain ⟨actual,other,ha,ho,hm⟩ := FiniteGrid.source M D (request a b) initial tensor launch
      (List.range (programs (request a b))) (fun _ hp => List.mem_range.mp hp) initial initial rfl
    have eq : other=typed := Option.some.inj (ho.symm.trans ht)
    subst other
    refine ⟨actual,typed,ha,ht,hm,?_,?_⟩
    · intro row hr col hc; rw [hm]; exact result.1 row hr col hc
    · intro region offset outside; rw [hm]; exact result.2 region offset outside

theorem safety {α : Type} [Inhabited α] (read : Storage → Nat → α) (a b c : Tensor)
    (h : InputContract a.metadata b.metadata) (nonempty : ¬Contract.Empty a.metadata b.metadata)
    (initial : State α) (output : Output a b c) (bound : KernelContract.Inputs read a b initial) :
    ∃ tensor : TensorAPI (request a b) initial,
      AddressSafety (request a b) initial tensor ∧ Index.Correspondence (request a b) initial tensor := by
  let tensor := tensorAPI read a b c h nonempty initial output bound
  exact ⟨tensor,address_safety _ _ tensor,Index.correspondence _ _ tensor
    (execution_nonempty _ _ h nonempty).1.2.2.2⟩

theorem post {α : Type} [Inhabited α] (Q : Numerics α) (a b c : Tensor)
    (initial final : State α) (output : Output a b c) (result : Result Q (request a b) initial final)
    (value : OutputIndex a b → α)
    (expected : ∀ i,reference Q (request a b) initial i.1.val i.2.val=value i) :
    CheckedHost.Post a b c initial final value := by
  refine ⟨output.metadata,?_,?_⟩
  · intro i
    have hv := result.1 i.1.val i.1.isLt i.2.val i.2.isLt
    rw [expected i] at hv
    have hn := i.2.isLt
    simpa only [request,outputOffset,address,max_eq_left (show 1≤columns b.metadata by omega),Nat.mul_comm] using hv
  · intro r o outside
    apply result.2 r o
    rintro ⟨hr,ho⟩
    rcases outside with ne | large
    · exact ne hr
    · obtain ⟨row,hr,col,hcol,rfl⟩ := ho
      have ho := linear_bound row col (rows a.metadata) (columns b.metadata) hr hcol
      have pos : 1≤columns b.metadata := by change col<columns b.metadata at hcol; omega
      simp only [outputOffset,request,max_eq_left pos] at large
      rw [output_numel a b c output.metadata] at large
      nlinarith

#axiomsClean empty_programs
#axiomsClean executes
#axiomsClean safety
#axiomsClean post
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource
