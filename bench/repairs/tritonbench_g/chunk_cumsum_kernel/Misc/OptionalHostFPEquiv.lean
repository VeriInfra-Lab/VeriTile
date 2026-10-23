import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.WholeFP

/-! Actual i64 source versus the complete logical-index reference source.
Configuration-specific FP32 primitives, arbitrary words, unspecified loads and
all operation ordering remain explicit; no FP-to-Real equality is asserted. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.OptionalHostFPEquiv
open VeriTile.Triton VeriTile.Torch Layout DirectHost FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def defaultPrecision : ComputeDType := .fp32
inductive NumericalSite where
  | literal | loadConversion | forwardScan | outputAdd | reduction | carryAdd | storeConversion
  deriving DecidableEq
def numericalSites : List NumericalSite := [.literal,.loadConversion,.forwardScan,.outputAdd,
  .reduction,.carryAdd,.storeConversion]

theorem wrapper_fp32 (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (c : Shape) (p : Storage.Placement c) (binding : TensorBinding s c p)
    (ops : Config → FPRow.Ops α) (padding : WholeFP.Padding α) (api : API (Cell α))
    (trials : List Config) (selected : Config) (caller : Heap (Cell α)) (live : Nat → Prop)
    (logical : Nat → Nat → α)
    (reads : ∀ row time, row<c.B*c.H → time<c.T →
      caller (Storage.address p.backing.input (row*c.T+time))=Cell.mk .fp32 (logical row time))
    (a : Allocated (Cell α)) (allocated : api.emptyLike caller=.ok a)
    (laws : AllocationLaws s c p caller a live) (device : api.launchDevice=s.device)
    (setups : c.B*c.H*c.T≠0 → Setups api 0 (trials++[selected])) :
    ∃ result reference,
      guarded s dtype c p (FPGrid.source ops c p.input p.output) (WholeFP.entry c p padding) State.mem
        api trials selected caller=.ok {a with heap:=result} ∧
      metadata s c p {a with heap:=result} ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        (WholeFP.entry c p padding a.heap).mem p.input (row*c.T+time)=Cell.mk .fp32 (logical row time)) ∧
      Host.sourceLaunches (FPGrid.referenceSource ops c p.input p.output) (trials++[selected])
        (WholeFP.entry c p padding a.heap)=some reference ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        result (Storage.address p.backing.output (row*c.T+time))=reference.mem p.output (row*c.T+time)) ∧
      (∀ off, off<p.backing.input.cells → result (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)) ∧
      (∀ addr, live addr → result addr=caller addr) ∧
      (∀ addr, ¬allocationWrites c p addr → result addr=caller addr) ∧
      (∀ addr, ¬Storage.writes c p addr → result addr=a.heap addr) ∧
      api.launchDevice=a.tensor.device := by
  rw [guarded_eq s dtype contract]
  have resultDevice := device.trans laws.metadata.2.2.2.2.1.symm
  have input : ∀ row time, row<c.B*c.H → time<c.T →
      (WholeFP.entry c p padding a.heap).mem p.input (row*c.T+time)=Cell.mk .fp32 (logical row time) := by
    intro row time hr ht
    simp only [WholeFP.entry,Storage.lift_input]
    rw [laws.input _ ((Storage.offset_bound c row time hr ht).trans_le p.backing.inputCells)]
    exact reads row time hr ht
  obtain ⟨t,reference,he,hr,out,frame⟩ := WholeFP.physical ops c p padding trials selected a.heap
  by_cases empty : c.B*c.H*c.T=0
  · refine ⟨a.heap,reference,?_,laws.metadata,input,hr,?_,laws.input,laws.liveFrame,laws.frame,fun _ _ => rfl,resultDevice⟩
    · simp [DirectHost.run,allocated,empty]
    · intro row time hrow htime
      have positive := Nat.mul_pos (Nat.zero_lt_of_lt hrow) (Nat.zero_lt_of_lt htime)
      omega
  · refine ⟨Storage.commit c p a.heap t.mem,reference,
      successful_return c p (FPGrid.source ops c p.input p.output) (WholeFP.entry c p padding) State.mem
        api trials selected caller a allocated empty t he (setups empty),laws.metadata,input,hr,out,?_,?_,?_,frame,resultDevice⟩
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

#axiomsClean wrapper_fp32
#print axioms wrapper_fp32
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.OptionalHostFPEquiv
