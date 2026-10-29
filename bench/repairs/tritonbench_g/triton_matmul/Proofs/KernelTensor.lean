import bench.repairs.tritonbench_g.triton_matmul.Misc.KernelCorrect
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelTensor
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter KernelContract KernelRealModel
noncomputable section

/-- Interpretation of C's completed finite stores at physical element-start
addresses. Arbitrary Reals have no byte encoding. Non-C addresses retain their
original values, including both original input views. -/
def realRead (read : KernelRealInputs.RealRead) (st : Storage) (c : Tensor) (final : State ℝ) : KernelRealInputs.RealRead :=
  fun ptr => if c.metadata.dataPtr.toNat≤ptr ∧ ptr<c.metadata.dataPtr.toNat+width st*c.metadata.numel ∧
    (ptr-c.metadata.dataPtr.toNat)%width st=0 then
      FPValue.read st (final.mem outputC ((ptr-c.metadata.dataPtr.toNat)/width st))
    else read ptr

theorem real_at (read : KernelRealInputs.RealRead) (st : Storage) (c : Tensor) (final : State ℝ)
    (i : Nat) (hi : i<c.metadata.numel) :
    realRead read st c final (c.metadata.dataPtr.toNat+width st*i)=FPValue.read st (final.mem outputC i) := by
  have inside : c.metadata.dataPtr.toNat≤c.metadata.dataPtr.toNat+width st*i ∧
      c.metadata.dataPtr.toNat+width st*i<c.metadata.dataPtr.toNat+width st*c.metadata.numel ∧
      (c.metadata.dataPtr.toNat+width st*i-c.metadata.dataPtr.toNat)%width st=0 := by
    cases st <;> simp only [width] at * <;> omega
  have index : (c.metadata.dataPtr.toNat+width st*i-c.metadata.dataPtr.toNat)/width st=i := by
    cases st <;> simp only [width] at * <;> omega
  simp only [realRead,if_pos inside,index]

theorem real_frame (read : KernelRealInputs.RealRead) (st : Storage) (c : Tensor) (final : State ℝ)
    (ptr : Nat) (outside : ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width st*c.metadata.numel≤ptr) :
    realRead read st c final ptr=read ptr := by
  have no : ¬(c.metadata.dataPtr.toNat≤ptr ∧ ptr<c.metadata.dataPtr.toNat+width st*c.metadata.numel ∧
      (ptr-c.metadata.dataPtr.toNat)%width st=0) := by omega
  simp only [realRead,if_neg no]

structure RealPost (read final : KernelRealInputs.RealRead) (a b c : Tensor) (value : OutputIndex a b → ℝ) : Prop where
  metadata : Metadata a b c
  logical : ∀ i,KernelRealInputs.logicalValue final c (address a b i)=value i
  frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨
    c.metadata.dataPtr.toNat+width (storage a.metadata)*c.metadata.numel≤ptr → final ptr=read ptr

theorem real_post (read : KernelRealInputs.RealRead) (a b c : Tensor) (initial final : State ℝ)
    (output : Output a b c) (value : OutputIndex a b → ℝ)
    (post : CheckedHost.Post a b c initial final value) :
    RealPost read (realRead read (storage a.metadata) c final) a b c value := by
  refine ⟨output.metadata,?_,real_frame read _ c final⟩
  intro i
  have hi : address a b i<c.metadata.numel := by
    rw [output_numel a b c output.metadata]
    exact linear_bound _ _ _ _ i.1.isLt i.2.isLt
  have st : storage c.metadata=storage a.metadata := same_storage a.metadata c.metadata output.metadata.2.1
  have ptr : physicalAddress (storage a.metadata) c (address a b i)=
      c.metadata.dataPtr.toNat+width (storage a.metadata)*address a b i := by
    rw [output.tensor.pointer (by omega)]
    simp only [physicalAddress,output.tensor.contiguous output.metadata.2.2.2.1 _ hi,Nat.mul_add,Nat.add_assoc]
  simp only [KernelRealInputs.logicalValue,st,ptr,real_at read _ c final _ hi,post.values i,read_stored]

specification real_tensor (read : KernelRealInputs.RealRead) (a b c : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State ℝ)
    (output : Output a b c) (bound : KernelRealInputs.Inputs read a b initial) :
    VeriTile.Spec.Real (∃ actual typed,
      FiniteGrid.grid ideal idealDot (request a b) (List.range (programs (request a b))) initial=some actual ∧
      runPrograms ideal idealDot (request a b) (List.range (programs (request a b))) initial=some typed ∧
      actual.mem=typed.mem ∧ CheckedHost.Post a b c initial actual (KernelCorrect.expected read a b) ∧
      RealPost read (realRead read (storage a.metadata) c actual) a b c (KernelCorrect.expected read a b)) := by
  obtain ⟨actual,typed,ha,ht,hm,post⟩ := KernelCorrect.kernel_real read a b c la lb h initial output bound
  exact ⟨actual,typed,ha,ht,hm,post,real_post read a b c initial actual output _ post⟩

#axiomsClean real_at
#axiomsClean real_frame
#axiomsClean real_post
#axiomsClean real_tensor
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelTensor
