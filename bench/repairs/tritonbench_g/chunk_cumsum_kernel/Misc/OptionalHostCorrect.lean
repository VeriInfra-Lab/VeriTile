import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.WholeReal

/-! Checked direct-input whole-call correctness. The independent Real target is
an ordinary inclusive prefix; source casts and rounding are erased here. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.OptionalHostCorrect
open VeriTile.Triton VeriTile.Torch Layout DirectHost
open scoped BigOperators
noncomputable section

abbrev logicalPrefix := WholeReal.expected

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype=.ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

theorem wrapper_correct (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (c : Shape) (p : Storage.Placement c) (binding : TensorBinding s c p)
    (padding : RegionName → Nat → ℝ) (api : API MemCell) (trials : List Config) (selected : Config)
    (caller : Heap MemCell) (live : Nat → Prop) (logical : Nat → Nat → ℝ)
    (reads : ∀ row time, row<c.B*c.H → time<c.T →
      WholeReal.read (caller (Storage.address p.backing.input (row*c.T+time)))=logical row time)
    (a : Allocated MemCell) (allocated : api.emptyLike caller=.ok a)
    (laws : AllocationLaws s c p caller a live) (device : api.launchDevice=s.device)
    (setups : c.B*c.H*c.T≠0 → Setups api 0 (trials++[selected])) :
    ∃ result, guarded s dtype c p (WholeReal.source c p) (WholeReal.entry c p padding) BlockState.mem
        api trials selected caller=.ok {a with heap:=result} ∧
      metadata s c p {a with heap:=result} ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        WholeReal.read (result (Storage.address p.backing.output (row*c.T+time)))=logicalPrefix logical c.T row time) ∧
      (∀ off, off<p.backing.input.cells → result (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)) ∧
      (∀ addr, live addr → result addr=caller addr) ∧
      (∀ addr, ¬allocationWrites c p addr → result addr=caller addr) ∧
      (∀ addr, ¬Storage.writes c p addr → result addr=a.heap addr) ∧
      api.launchDevice=a.tensor.device := by
  rw [guarded_eq s dtype contract]
  have resultDevice := device.trans laws.metadata.2.2.2.2.1.symm
  by_cases empty : c.B*c.H*c.T=0
  · refine ⟨a.heap,?_,laws.metadata,?_,laws.input,laws.liveFrame,laws.frame,fun _ _ => rfl,resultDevice⟩
    · simp [run,allocated,empty]
    · intro row time hr ht
      have positive := Nat.mul_pos (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt ht)
      omega
  · obtain ⟨t,he,out,frame⟩ := WholeReal.physical c p padding trials selected a.heap
    refine ⟨Storage.commit c p a.heap t.mem,
      successful_return c p (WholeReal.source c p) (WholeReal.entry c p padding) BlockState.mem
        api trials selected caller a allocated empty t he (setups empty),laws.metadata,?_,?_,?_,?_,frame,resultDevice⟩
    · intro row time hr ht
      rw [out row time hr ht]
      apply Finset.sum_congr rfl
      intro k hk
      have hk' := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
      dsimp only
      rw [laws.input _ ((Storage.offset_bound c row k hr hk').trans_le p.backing.inputCells)]
      exact reads row k hr hk'
    · intro off inside
      exact (frame _ (Storage.input_not_written c p off inside)).trans (laws.input off inside)
    · intro addr hl
      exact (frame addr (laws.fresh addr hl)).trans (laws.liveFrame addr hl)
    · intro addr outside
      exact (frame addr (fun written => outside (writes_within_allocation c p addr written))).trans (laws.frame addr outside)

abbrev native_pid := TensorBinding.pid
abbrev native_loop := NativeIndex.loop_native
abbrev native_indices := NativeIndex.source_indices
abbrev native_pointers := NativeIndex.active_pointers
abbrev original_overflow := NativeIndex.original_row_overflow

#axiomsClean python_checks_iff_preconditions
#axiomsClean wrapper_correct
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.OptionalHostCorrect
