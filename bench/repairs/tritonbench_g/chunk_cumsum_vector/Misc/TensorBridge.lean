import bench.repairs.tritonbench_g.chunk_cumsum_vector.Misc.KernelRealInterface
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Misc.KernelFPInterface
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.KernelContract

/-! Optional input checks and Tensor/native address bindings. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.TensorBridge
open VeriTile.Triton VeriTile.Torch
noncomputable section

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype = .ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

abbrev native_pid := TensorBinding.pid

theorem tensor_input_memory {β γ : Type} (s : TensorMetadata) (h : Call)
    (p : Storage.Placement h) (binding : TensorBinding s h p)
    (other : β) (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (law : InputReadLaw s h heap read logical)
    (row time channel : Nat) (hr : row<h.B*h.H) (ht : time<h.T) (hc : channel<h.S) :
    read (Storage.lift h p other heap h.input
      (h.inputStrides.address h.H row time channel))=logical row time channel :=
  TensorBinding.logical_input_read s h p binding other heap read logical law row time channel hr ht hc

open NativeIndex

theorem native_addresses (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    let time := Geometry.time h c q i.val
    let channel := Geometry.column h p.feature j.val
    (offsetWord h p h.inputStrides time channel).toInt=(h.inputStrides.address h.H p.row time channel:Int) ∧
    (offsetWord h p h.outputStrides time channel).toInt=(h.outputStrides.address h.H p.row time channel:Int) ∧
    (BitVec.ofNat 64 backing.input.base+offsetWord h p h.inputStrides time channel*BitVec.ofNat 64 backing.input.bytes).toNat=
      backing.input.base+h.inputStrides.address h.H p.row time channel*backing.input.bytes ∧
    (BitVec.ofNat 64 backing.output.base+offsetWord h p h.outputStrides time channel*BitVec.ofNat 64 backing.output.bytes).toNat=
      backing.output.base+h.outputStrides.address h.H p.row time channel*backing.output.bytes :=
  NativeIndex.source_addresses h c hc backing p q hq i j

theorem native_coordinates (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    coordinateWord h.T (q*c.BT) i.val=BitVec.ofNat 64 (Geometry.time h c q i.val) ∧
    coordinateWord h.S (p.feature*BS) j.val=BitVec.ofNat 64 (Geometry.column h p.feature j.val) :=
  NativeIndex.source_coordinates h c hc backing p q hq i j

theorem native_indices (h : Call) (c : Config) (hc : c∈configurations)
    (backing : Backing h) (p : PID h) (q : Nat) (hq : q<Geometry.chunks h c)
    (i : Fin c.BT) (j : Fin BS) :
    Geometry.chunks h c<signedLimit ∧
    (BitVec.ofNat 64 q*BitVec.ofNat 64 c.BT).toInt=(q*c.BT:Int) ∧
    (BitVec.ofInt 64 (BitVec.ofNat 32 p.feature).toInt*BitVec.ofNat 64 BS).toInt=(p.feature*BS:Int) ∧
    (BitVec.ofNat 64 h.T-BitVec.ofNat 64 (q*c.BT)).toInt=(h.T-q*c.BT:Nat) ∧
    (BitVec.ofNat 64 h.S-BitVec.ofNat 64 (p.feature*BS)).toInt=(h.S-p.feature*BS:Nat) ∧
    ((BitVec.ofNat 64 i.val).slt (BitVec.ofNat 64 (h.T-q*c.BT))=decide (Geometry.activeTime h c q i.val)) ∧
    ((BitVec.ofNat 64 j.val).slt (BitVec.ofNat 64 (h.S-p.feature*BS))=decide (Geometry.activeColumn h p.feature j.val)) :=
  NativeIndex.source_indices h c hc backing p q hq i j

#axiomsClean python_checks_iff_preconditions
#axiomsClean tensor_input_memory
#axiomsClean native_addresses
#axiomsClean native_coordinates
#axiomsClean native_indices

namespace Correct
open VeriTile.Bench.Repairs.ChunkCumsumVector.RealInterface
/-- The physical cells consumed by this source are the admitted Tensor's
logical input cells, through the same representation law used by FPEquiv. -/
theorem kernel_correct (tensor : TensorMetadata) (h : Call)
    (p : Storage.Placement h) (binding : TensorBinding tensor h p)
    (padding : Config → Geometry.Program h → RegionName → Nat → ℝ)
    (c : Config) (chosen : c∈configurations) (program : Geometry.Program h)
    (before : RealStorage.Heap) (logical : Nat → Nat → Nat → ℝ)
    (reads : InputReadLaw tensor h before RealStorage.read logical) :
    Spec.Real (∃ after, RealStorage.execute h p padding c program before=some after ∧
      Stream.StableInput h p before after ∧
      (∀ time j, Grid.active h program time j →
        RealStorage.read (after (Storage.address p.backing.output (Grid.offset h program time j)))=
          mathematical (fun k => logical program.row k (program.feature*BS+j.val)) h.T time) ∧
      (∀ addr, ¬Storage.writes h p [program] addr → after addr=before addr)) := by
  obtain ⟨after,executed,stable,values,frame⟩ :=
    memory_correct h (binding.outputLayout) p padding c chosen program before
  refine ⟨after,executed,stable,?_,frame⟩
  intro time j active
  rw [values time j active]
  unfold mathematical
  apply Finset.sum_congr rfl
  intro k hk
  have kt := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
  exact TensorBinding.physical_input_read tensor h p binding before RealStorage.read
    logical reads _ _ _ program.rowInside kt active.2
#axiomsClean kernel_correct
end Correct

namespace FPEquiv
open VeriTile Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.ChunkCumsumVector.FPInterface
variable {α : Type} [Inhabited α]
/-- Both actual sources start from the Tensor's current physical heap and
preserve its logical FP32 input values, including nonfinite input payloads. -/
theorem chunk_sum_left_equiv (M : Algebra α) (dot : DotBackend α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (tensor : Torch.TensorMetadata) (h : Call) (storage : Storage.Placement h)
    (binding : TensorBinding tensor h storage) (heap : Nat → Cell α)
    (logical : Nat → Nat → Nat → α)
    (reads : InputReadLaw tensor h heap (Cell.read .fp32) logical)
    (c : Config) (chosen : c∈configurations) (p : Geometry.Program h)
    (u : State α) (other : Cell α) (memory : u.mem=Storage.lift h storage other heap)
    (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row)
    (finite : rewrite_domain M D h c p u) :
    ∃ a b, FP.Structural.exec (machine M dot) (original h c) u=some a ∧
      FP.Structural.exec (machine M dot) (sumLeft h c) u=some b ∧
      a.mem=b.mem ∧ FPMemory.Frame h p u a ∧ FPMemory.Frame h p u b ∧
      (∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
        (a.mem h.input (h.inputStrides.address h.H row time channel)).read .fp32=
          logical row time channel ∧
        (b.mem h.input (h.inputStrides.address h.H row time channel)).read .fp32=
          logical row time channel) := by
  obtain ⟨a,b,ha,hb,equal,fa,fb⟩ :=
    memory_equiv M dot D model h binding.outputLayout c chosen p u hf hr finite
  refine ⟨a,b,ha,hb,equal,fa,fb,?_⟩
  intro row time channel hr ht hc
  have input := TensorBinding.logical_input_read tensor h storage binding other heap
    (Cell.read .fp32) logical reads row time channel hr ht hc
  have outside (off : Nat) : ¬FPMemory.footprint h p (h.input,off) :=
    fun written => binding.distinct written.1
  rw [← fa h.input _ (outside _),← fb h.input _ (outside _),memory]
  exact ⟨input,input⟩
#axiomsClean chunk_sum_left_equiv
end FPEquiv
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.TensorBridge
