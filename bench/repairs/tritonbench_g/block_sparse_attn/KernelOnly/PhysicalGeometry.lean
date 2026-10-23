import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Contract
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Readback

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSectionVars Bool.false
namespace PhysicalKernel
variable {α : Type} [Inhabited α]

def shape (p : Parameters) (batches : Nat) : Shape := (layoutOf p batches (fun _ => 0) (fun _ => 0)).shape

def coordinate (p : Parameters) (batches : Nat) (id : Program) (i : TileIndex [16,16]) : Work :=
  laneWork (shape p batches) (id.lane i)

def programs (p : Parameters) (batches : Nat) : List Program :=
  (List.range (ceil p.length 16)).flatMap (fun r =>
    (List.range (batches*p.heads)).flatMap (fun h =>
      (List.range p.blocks).map (fun c => ⟨r,h,c⟩)))

theorem programs_mem (p : Parameters) (batches : Nat) (id : Program) :
    id∈programs p batches ↔ id.Valid p batches := by
  rcases id with ⟨r,h,c⟩
  simp [programs,Program.Valid,Program.mk.injEq]

theorem state_work (access : Access α Store Byte) (zero : α) (buffers : Physical.Buffers)
    (m : Store) (p : Parameters) (batches : Nat) (id : Program) (i : TileIndex [16,16]) :
    work p (state access zero buffers m id) i=coordinate p batches id i := rfl

theorem coordinate_active (p : Parameters) (batches : Nat) (id : Program)
    (valid : id.Valid p batches) (heads : 0<p.heads) (i : TileIndex [16,16])
    (live : id.row*16+i.1.val<p.length) : (coordinate p batches id i).active (shape p batches) := by
  exact (Source.store_mask (shape p batches) heads (id.lane i)
    (lane_valid p batches _ _ id valid i)).mpr live

theorem coordinate_coverage (p : Parameters) (batches : Nat) (w : Work)
    (live : w.active (shape p batches)) :
    ∃ id∈programs p batches,∃ i : TileIndex [16,16],
      id.row*16+i.1.val<p.length ∧ coordinate p batches id i=w := by
  obtain ⟨l,valid,equal⟩ := lane_coverage (shape p batches)
    (by norm_num [shape,layoutOf]) (by norm_num [shape,layoutOf]) w live
  let id : Program := ⟨l.rowProgram,l.headProgram,l.channelProgram⟩
  let i : TileIndex [16,16] := (⟨l.rowLane,valid.2.2.2.1⟩,⟨l.colLane,valid.2.2.2.2⟩,PUnit.unit)
  refine ⟨id,(programs_mem p batches id).mpr ⟨valid.1,valid.2.1,valid.2.2.1⟩,i,?_,equal⟩
  have rows := congrArg Work.row equal
  change l.rowProgram*16+l.rowLane=w.row at rows
  change l.rowProgram*16+l.rowLane<p.length
  rw [rows]
  exact live.2.2.1

theorem output_offset (access : Access α Store Byte) (zero : α) (buffers : Physical.Buffers)
    (m : Store) (p : Parameters) (batches : Nat) (id : Program) (i : TileIndex [16,16])
    (parameters : p.ob=p.heads*p.length*(16*p.blocks) ∧ p.oh=p.length*(16*p.blocks) ∧ p.om=16*p.blocks) :
    outputOffset p (state access zero buffers m id) i=(coordinate p batches id i).target (shape p batches) := by
  simp only [outputOffset,coordinate,shape,layoutOf,work,Work.target,outputAddress,
    state,Program.pids,Program.lane,laneWork,batch,head,row,Shape.dim,
    parameters.1,parameters.2.1,parameters.2.2]
  ring

theorem abi_stride (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (m : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers m q k v out rows cols) : 16≤p.om := by
  have positive : 0<p.blocks := abi.header.shape.2.2.2.2.2.2.2.2.1
  rw [abi.parameters.2.2]
  omega

theorem output_window (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (m : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers m q k v out rows cols)
    (w : Work) (live : w.active (shape p batches)) :
    buffers.output.base≤Physical.outputAddress buffers (shape p batches) w ∧
    Physical.outputAddress buffers (shape p batches) w+4≤buffers.output.base+buffers.output.storage.bytes := by
  have bound := target_bound (shape p batches) w live
  have extent := abi.outputExtent
  rw [← abi.outputStorage] at extent
  change w.target (shape p batches)<batches*p.heads*p.length*(16*p.blocks) at bound
  unfold Physical.outputAddress Physical.inputAddress
  omega

theorem output_word (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (m : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers m q k v out rows cols)
    (w : Work) (live : w.active (shape p batches)) (j : Fin 4) :
    Physical.outputAddress buffers (shape p batches) w+j.val<2^64 ∧
    (BitVec.ofNat 64 (Physical.outputAddress buffers (shape p batches) w+j.val)).toNat=
      Physical.outputAddress buffers (shape p batches) w+j.val := by
  have bound := (output_window p batches access buffers m q k v out rows cols abi w live).2
  have backing := abi.backing.2
  have valid := j.isLt
  have lt : Physical.outputAddress buffers (shape p batches) w+j.val<2^64 := by omega
  exact ⟨lt,by change _%2^64=_; exact Nat.mod_eq_of_lt lt⟩

end PhysicalKernel
#axiomsClean PhysicalKernel.programs_mem
#axiomsClean PhysicalKernel.coordinate_active
#axiomsClean PhysicalKernel.coordinate_coverage
#axiomsClean PhysicalKernel.output_offset
#axiomsClean PhysicalKernel.abi_stride
#axiomsClean PhysicalKernel.output_window
#axiomsClean PhysicalKernel.output_word
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
