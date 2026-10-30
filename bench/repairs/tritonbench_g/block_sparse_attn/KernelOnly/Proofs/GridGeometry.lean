import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.IO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

abbrev OutIndex (p : Parameters) (batches : Nat) :=
  TileIndex [ceil p.length 16,batches*p.heads,p.blocks,16,16]

def programOf {p : Parameters} {batches : Nat} (j : OutIndex p batches) : PhysicalKernel.Program :=
  ⟨j.1.val,j.2.1.val,j.2.2.1.val⟩
def laneOf {p : Parameters} {batches : Nat} (j : OutIndex p batches) : TileIndex [16,16] :=
  (j.2.2.2.1,j.2.2.2.2.1,PUnit.unit)
def live {p : Parameters} {batches : Nat} (j : OutIndex p batches) : Prop :=
  j.1.val*16+j.2.2.2.1.val<p.length

theorem program_valid {p : Parameters} {batches : Nat} (j : OutIndex p batches) :
    (programOf j).Valid p batches := ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt⟩

theorem output_target (p : Parameters) (batches : Nat) (geometry : Geometry p)
    (j : OutIndex p batches) : outputOffset p batches j=
      (PhysicalKernel.coordinate p batches (programOf j) (laneOf j)).target (PhysicalKernel.shape p batches) := by
  simp only [outputOffset,PhysicalKernel.coordinate,PhysicalKernel.shape,layoutOf,programOf,laneOf,
    PhysicalKernel.Program.lane,laneWork,Work.target,BlockSparseAttn.outputAddress,Shape.dim,
    geometry.2.2.1,geometry.2.2.2.1,geometry.2.2.2.2,batch,head]
  ring

theorem output_injective (p : Parameters) (batches : Nat) (geometry : Geometry p)
    (i j : OutIndex p batches) (hi : live i) (hj : live j)
    (equal : outputOffset p batches i=outputOffset p batches j) : i=j := by
  rw [output_target p batches geometry i,output_target p batches geometry j] at equal
  have workeq := target_injective (PhysicalKernel.shape p batches) _ _
    (PhysicalKernel.coordinate_active p batches (programOf i) (program_valid i) geometry.1 (laneOf i) hi)
    (PhysicalKernel.coordinate_active p batches (programOf j) (program_valid j) geometry.1 (laneOf j) hj) equal
  have laneeq := lane_injective (PhysicalKernel.shape p batches) _ _
    (PhysicalKernel.lane_valid p batches (fun _=>0) (fun _=>0) (programOf i) (program_valid i) (laneOf i))
    (PhysicalKernel.lane_valid p batches (fun _=>0) (fun _=>0) (programOf j) (program_valid j) (laneOf j)) workeq
  rcases i with ⟨ir,ih,ic,im,id,⟨⟩⟩
  rcases j with ⟨jr,jh,jc,jm,jd,⟨⟩⟩
  simp only [programOf,laneOf,PhysicalKernel.Program.lane,Lane.mk.injEq] at laneeq
  simp only [Prod.mk.injEq,Fin.ext_iff]
  exact ⟨laneeq.1,laneeq.2.1,laneeq.2.2.1,laneeq.2.2.2.1,laneeq.2.2.2.2,True.intro⟩

theorem source_offset {α : Type} (p : Parameters) (batches : Nat)
    (s : State α) (j : OutIndex p batches) (ids : s.pids=(programOf j).pids) :
    KernelOnly.outputOffset p s (laneOf j)=outputOffset p batches j := by
  simp only [KernelOnly.outputOffset,KernelOnly.batch,KernelOnly.head,KernelOnly.row,
    ids,programOf,laneOf,PhysicalKernel.Program.pids,outputOffset,batch,head]

end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
