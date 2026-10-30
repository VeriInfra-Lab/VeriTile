import bench.repairs.tritonbench_g.triton_matmul.Proofs.InputAdapter
import bench.repairs.tritonbench_g.triton_matmul.Proofs.FP
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.CheckedHost
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter
noncomputable section
set_option maxHeartbeats 6000000

def call {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (allocate : Allocator α)
    (a b : Tensor) (before : State α) : Option (Tensor × State α) :=
  if checkTensorInputs a.metadata b.metadata=.ok () then
    (allocate a b before).bind fun (c,prepared) =>
      if Contract.Empty a.metadata b.metadata then some (c,prepared)
      else (host M D (request a b) prepared).toOption.map (fun result => (c,result.2))
  else none

def left {α : Type} (read : Read) (decode : TensorBinding.Word → α) (a b : Tensor)
    (la : Laws (storage a.metadata) a) : Fin (rows a.metadata) → Fin (columns a.metadata) → α :=
  fun r k => decode (logicalValue read (storage a.metadata) a la (r.val*columns a.metadata+k.val))
def right {α : Type} (read : Read) (decode : TensorBinding.Word → α) (a b : Tensor)
    (lb : Laws (storage b.metadata) b) : Fin (columns a.metadata) → Fin (columns b.metadata) → α :=
  fun k col => decode (logicalValue read (storage b.metadata) b lb (k.val*columns b.metadata+col.val))

theorem source_inputs {α : Type} [Inhabited α] (read : Read) (decode : TensorBinding.Word → α)
    (a b : Tensor) (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State α) (bound : Inputs read decode a b s) :
    (∀ r k,Repair.inputA (request a b) s r.val k.val=left read decode a b la r k) ∧
    (∀ k col,Repair.inputB (request a b) s k.val col.val=right read decode a b lb k col) := by
  constructor
  · intro r k
    have hi : r.val*columns a.metadata+k.val<a.metadata.numel := by
      rw [matrix_numel a.metadata h.rank.1]
      exact linear_bound _ _ _ _ r.isLt k.isLt
    simpa only [Repair.inputA,request,left,Nat.mul_one] using logical_left read decode a b la h s bound _ hi
  · intro k col
    have hi : k.val*columns b.metadata+col.val<b.metadata.numel := by
      rw [matrix_numel b.metadata h.rank.2,←h.inner]
      exact linear_bound _ _ _ _ k.isLt col.isLt
    simpa only [Repair.inputB,request,right,Nat.mul_one] using logical_right read decode a b lb h s bound _ hi

structure Post {α : Type} (a b c : Tensor) (before final : State α)
    (value : OutputIndex a b → α) : Prop where
  metadata : Metadata a b c
  values : ∀ i,final.mem outputC (address a b i)=stored (storage a.metadata) (value i)
  frame : ∀ r o,(r≠outputC ∨ c.metadata.numel≤o) → final.mem r o=before.mem r o

/-- Complete checked host with ordinary fresh allocation, original logical
input binding, whole source/grid execution, returned object and frame. -/
theorem correct {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (Q : Numerics α)
    (allocate : Allocator α) (read : Read) (decode : TensorBinding.Word → α) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata)
    (laws : PrimitiveLaws M D Q (storage a.metadata)) (before : State α)
    (allocates : AllocationLaw allocate a b before) (bound : Inputs read decode a b before) :
    ∃ c prepared final,allocate a b before=some (c,prepared) ∧
      call M D allocate a b before=some (c,final) ∧
      Post a b c before final (fun i => logicalReference Q (request a b)
        (left read decode a b la) (right read decode a b lb) i.1 i.2) ∧
      (¬Contract.Empty a.metadata b.metadata → ∃ tensor : TensorAPI (request a b) prepared,
        host M D (request a b) prepared=.ok (outputC,final) ∧
        AddressSafety (request a b) prepared tensor ∧ Index.Correspondence (request a b) prepared tensor) := by
  obtain ⟨c,prepared,allocated,allocation⟩ := allocates
  have checked := (checkTensorInputs_iff a.metadata b.metadata).mpr h
  by_cases empty : Contract.Empty a.metadata b.metadata
  · refine ⟨c,prepared,prepared,allocated,?_,⟨allocation.metadata,?_,allocation.frame⟩,fun ne => False.elim (ne empty)⟩
    · simp [call,checked,allocated,empty]
    · intro i
      have ir := i.1.isLt; have ic := i.2.isLt
      rcases empty with e | e <;> omega
  · let tensor := tensorAPI read decode a b c h empty before prepared allocation bound
    have bounds := execution_nonempty a.metadata b.metadata h empty
    have launch : programs (request a b)<2^31 := by
      simpa only [programs,tileRows,tileColumns,request] using bounds.1.2.2.2
    have preparedInputs := allocated_inputs read decode a b c before prepared allocation bound
    have logical := source_inputs read decode a b la lb h prepared preparedInputs
    obtain ⟨final,he,values,frame⟩ := host_correct M D Q (request a b) laws prepared
      tensor.freshA tensor.freshB tensor.dimensions tensor.storage
    refine ⟨c,prepared,final,allocated,?_,⟨allocation.metadata,?_,?_⟩,?_⟩
    · simp [call,checked,allocated,empty,he,Except.toOption]
    · intro i
      have hv := values i.1.val i.1.isLt i.2.val i.2.isLt
      rw [reference_logical Q (request a b) prepared _ _ logical.1 logical.2 i.1 i.2] at hv
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

theorem rejected {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α) (allocate : Allocator α)
    (a b : Tensor) (before : State α) (h : ¬InputContract a.metadata b.metadata) :
    call M D allocate a b before=none := by
  simp [call,(checkTensorInputs_iff a.metadata b.metadata).not.mpr h]

#axiomsClean source_inputs
#axiomsClean correct
#axiomsClean rejected
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.CheckedHost
