import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Admission
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Runtime
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RmsnormFused.Binding
open VeriTile Triton FP.Structural Torch.FP32
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- The same Tensor object supplies metadata, storage, and the launch pointer. -/
structure Tensor where
  actual : Torch.FP32.Tensor
  laws : TensorLaws actual
  contiguous : actual.metadata.isContiguous = Bool.true
  notNeg : actual.metadata.isNeg = Bool.false
  conjugated : Bool

def Tensor.info (t : Tensor) : TensorInfo := ⟨t.actual.metadata,t.actual.storageOffset,t.conjugated⟩
def Tensor.view (t : Tensor) := dataView t.actual t.laws t.contiguous
def Tensor.size (t : Tensor) := t.actual.metadata.numel
def Tensor.read {α : Type} (t : Tensor) (memory : Nat → α) (i : Nat) :=
  memory (t.actual.storage.address (t.actual.physicalIndex i))

theorem Tensor.read_view (t : Tensor) (memory : Nat → α) (i : Nat) (hi : i<t.size) :
    t.read memory i=memory (t.view.address i) := by
  unfold Tensor.read
  rw [t.laws.contiguous t.contiguous i hi]
  congr 1
  have nonempty : t.actual.metadata.numel≠0 := by change t.size≠0; omega
  simp [Tensor.view,dataView,DataView.address,DataView.base,nonempty,Storage.address,Nat.mul_add,Nat.add_assoc]

theorem Tensor.read_logical (t : Tensor) (memory : Nat → Word) (i : Nat) :
    t.read memory i=logicalValue memory t.actual t.laws i := by
  simp [Tensor.read,logicalValue,t.notNeg]

/-- Ordinary kernel-entry output allocation facts, without any result/execution
premise. Read-only X and W may share storage or overlap arbitrarily. -/
structure Output (x w y : Tensor) : Prop where
  shape : y.actual.metadata.shape=x.actual.metadata.shape
  offset : y.actual.storageOffset=0
  device : y.actual.metadata.device=x.actual.metadata.device
  aligned : 0<y.size → y.actual.dataPtr.toNat%4=0
  inputFresh : y.view.base+4*y.size≤x.view.base ∨ x.view.base+4*x.size≤y.view.base
  weightFresh : y.view.base+4*y.size≤w.view.base ∨ w.view.base+4*w.size≤y.view.base

theorem Output.size (out : Output x w y) : y.size=x.size := by
  simp only [Tensor.size,Torch.TensorMetadata.numel,out.shape]

def initial {α : Type} [Inhabited α] (x w y : Tensor) (memory : Nat → α) : State α where
  mem := fun r i => .mk .real (memory ((if r="X" then x.view else if r="W" then w.view else y.view).address i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def footprint (y : Tensor) (address : Nat) : Prop :=
  y.view.base≤address ∧ address<y.view.base+4*y.size

/-- Complete physical word memory. Only valid output word addresses are updated. -/
def commit {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α) (s : State α) : Nat → α :=
  fun address => if footprint y address ∧ (address-y.view.base)%4=0 then
    Runtime.read s "Y" ((address-y.view.base)/4) else memory address

def reference {α : Type} (y : Tensor) (memory : Nat → α) (values : Nat → α) : Nat → α :=
  fun address => if footprint y address ∧ (address-y.view.base)%4=0 then
    values ((address-y.view.base)/4) else memory address

theorem commit_value {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (i : Nat) (hi : i<y.size) :
    commit y memory s (y.view.address i)=Runtime.read s "Y" i := by
  simp [commit,footprint,DataView.address,show 4*i<4*y.size by omega]

theorem commit_frame {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (address : Nat) (outside : ¬footprint y address) :
    commit y memory s address=memory address := by simp [commit,outside]

theorem commit_reference {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (values : Nat → α) (all : ∀ i,i<y.size → Runtime.read s "Y" i=values i) :
    commit y memory s=reference y memory values := by
  funext address
  simp only [commit,reference]
  split
  · rename_i active
    exact all _ (by have := active.1; unfold footprint at this; omega)
  · rfl

theorem input_sizes (x w : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps) :
    x.size=(geometry x.info).rows*width x.info ∧ w.size=width x.info := by
  constructor
  · exact (geometry_size x.info c.rank).symm
  · change w.info.numel=width x.info
    simp [Torch.TensorMetadata.numel,c.weightShape]

theorem flat_coordinates (x w : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (k : Nat) (active : k<x.size) :
    k/width x.info<(geometry x.info).rows ∧ k%width x.info<width x.info ∧
      (k/width x.info)*width x.info+k%width x.info=k := by
  refine ⟨?_,Nat.mod_lt _ c.widthPositive,?_⟩
  · apply (Nat.div_lt_iff_lt_mul c.widthPositive).mpr
    rw [←(input_sizes x w eps c).1]
    exact active
  · simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div k (width x.info)

/-- Successful actual DSL execution, all source cells and full region frame. -/
theorem execution {α : Type} [Inhabited α] (A : Algebra α)
    (x w y : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (argument : α) (memory : Nat → α) (schedule : List Nat)
    (all : schedule.Perm (geometry x.info).grid) :
    ∃ final, Runtime.run A "X" "Y" "W" (geometry x.info).rows (width x.info)
      (RmsnormFused.block (width x.info)) argument schedule (initial x w y memory)=some final ∧
      (∀ row col,row<(geometry x.info).rows → col<width x.info →
        Runtime.read final "Y" (row*width x.info+col)=
          Runtime.value A (initial x w y memory) "X" "W" row (width x.info)
            (RmsnormFused.block (width x.info)) argument col) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ x.size≤offset →
        final.mem region offset=(initial x w y memory).mem region offset) := by
  have valid := geometry_valid x.info w.info eps c
  obtain ⟨final,ran,hm⟩ := Runtime.source_complete A "X" "Y" "W"
    (geometry x.info).rows (width x.info) (RmsnormFused.block (width x.info)) argument
    c.widthPositive c.block schedule (initial x w y memory)
  refine ⟨final,ran,?_,?_⟩
  · intro row col hr hc
    unfold Runtime.read
    rw [hm]
    exact Runtime.whole_value A (geometry x.info) valid "X" "Y" "W" argument schedule all
      (initial x w y memory) row col hr hc (by decide) (by decide)
  · intro region offset outside
    rw [hm]
    apply Runtime.grid_frame
    rcases outside with different|outside
    · exact .inl different
    · right
      intro row hr col hc equal
      have bound := offset_bounded (geometry x.info) row col
        ((grid_complete _ _).mp (all.mem_iff.mp hr)) hc
      rw [geometry_size x.info c.rank] at bound
      change row*width x.info+col<x.size at bound
      omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Binding
