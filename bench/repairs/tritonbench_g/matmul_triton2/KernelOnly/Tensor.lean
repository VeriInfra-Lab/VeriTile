import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Model
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.TensorBinding
open VeriTile.Triton FP.Structural Model
open VeriTile.Torch.FP32
noncomputable section
set_option maxHeartbeats 8000000

theorem pointer_base (t : KernelOnly.Tensor) (laws : Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (positive : 0 < t.metadata.numel) :
    t.metadata.dataPtr.toNat=(dataView t laws contiguous).base := by
  have pointer := actual_pointer t laws contiguous positive
  have span := (dataView t laws contiguous).spanFits
  have bound : (dataView t laws contiguous).base < 2^64 := by
    change (dataView t laws contiguous).base+4*t.metadata.numel ≤ 2^64 at span
    omega
  simpa only [DataView.address,Nat.mul_zero,Nat.add_zero,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]
    using (congrArg BitVec.toNat pointer).symm

theorem logical_read (t : KernelOnly.Tensor) (laws : Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (noNeg : t.metadata.isNeg=Bool.false)
    (read : Nat → Word) (i : Nat) (hi : i < t.metadata.numel) :
    read (address t i)=logicalValue read t laws i := by
  rw [address,pointer_base t laws contiguous (by omega)]
  exact direct_read_eq_logical read t laws contiguous noNeg i hi

theorem output_numel (c : Arguments) (h : KernelContract c) :
    c.output.metadata.numel=rows c.a.metadata*columns c.b.metadata := by
  simp [VeriTile.Torch.TensorMetadata.numel,h.output.shape]

theorem output_index (c : Arguments) (h : KernelContract c) (i : Nat)
    (hi : i < rows c.a.metadata*columns c.b.metadata) : c.output.physicalIndex i=i := by
  have bound : i < c.output.metadata.numel := by rw [output_numel c h]; exact hi
  simpa only [h.output.offset,Nat.zero_add] using c.outputLaws.contiguous h.output.contiguous i bound

theorem output_pointer (c : Arguments) (h : KernelContract c)
    (positive : 0 < rows c.a.metadata*columns c.b.metadata) :
    c.output.metadata.dataPtr.toNat=c.output.storage.base := by
  have numel : 0 < c.output.metadata.numel := by rw [output_numel c h]; exact positive
  rw [pointer_base c.output c.outputLaws h.output.contiguous numel]
  simp [DataView.base,dataView,Nat.ne_of_gt numel,Storage.address,h.output.offset]

/-- Actual typed stores read back through the supplied output object's pointer.
Only output word addresses are projected; every other physical address keeps
the caller's value. This definition assumes no final-result equality. -/
def project {α : Type} [Inhabited α] (read : Nat → α) (c : Arguments) (after : State α)
    (a : Nat) : α :=
  if c.output.metadata.dataPtr.toNat ≤ a ∧
      a < c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) ∧
      (a-c.output.metadata.dataPtr.toNat)%4=0 then
    (after.mem "C" ((a-c.output.metadata.dataPtr.toNat)/4)).read .real else read a

theorem projected {α : Type} [Inhabited α] (read : Nat → α) (c : Arguments) (after : State α)
    (i : Nat) (hi : i < rows c.a.metadata*columns c.b.metadata) :
    project read c after (address c.output i)=(after.mem "C" i).read .real := by
  have inside : c.output.metadata.dataPtr.toNat+4*i <
      c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) := by omega
  have div : 4*i/4=i := by omega
  have mod : 4*i%4=0 := by omega
  simp only [project,address,Nat.le_add_right,inside,Nat.add_sub_cancel_left,
    div,mod,and_self,ite_true]

theorem exterior {α : Type} [Inhabited α] (read : Nat → α) (c : Arguments) (after : State α)
    (a : Nat) (outside : a < c.output.metadata.dataPtr.toNat ∨
      c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) ≤ a) :
    project read c after a=read a := by
  have miss : ¬(c.output.metadata.dataPtr.toNat ≤ a ∧
      a < c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) ∧
      (a-c.output.metadata.dataPtr.toNat)%4=0) := by omega
  simp only [project,miss,ite_false]

theorem input_frame {α : Type} [Inhabited α] (read : Nat → α) (c : Arguments)
    (h : KernelContract c) (after : State α) (t : KernelOnly.Tensor) (apart : apart c.output t)
    (a : Nat) (inside : t.storage.base ≤ a ∧ a < t.storage.base+4*t.storage.elements) :
    project read c after a=read a := by
  apply exterior
  by_cases positive : 0 < rows c.a.metadata*columns c.b.metadata
  · rw [output_pointer c h positive]
    have capacity := contiguous_capacity c.output c.outputLaws h.output.contiguous
      (by rw [output_numel c h]; exact positive)
    rw [output_numel c h] at capacity
    unfold KernelOnly.apart at apart
    omega
  · have empty : rows c.a.metadata*columns c.b.metadata=0 := by omega
    simp only [empty,Nat.mul_zero,Nat.add_zero]
    omega

#axiomsClean logical_read
#axiomsClean output_index
#axiomsClean output_pointer
#axiomsClean projected
#axiomsClean input_frame
#specNonCircular Arguments avoiding [Model.run,Revision001.Core.grid,FP.Structural.exec]
#specNonCircular project avoiding [Model.run,Revision001.Core.grid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.TensorBinding
