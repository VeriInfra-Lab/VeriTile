import bench.tritonbench_g.block_sparse_attn.Proofs.InputBindings
import bench.tritonbench_g.block_sparse_attn.Proofs.AccessFragments

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

theorem access_bounds (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
    (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) (bounds : RegionName→Nat)
    (bounded : (io a native encode observe p batches visits).InBounds bounds xs) :
    AccessBounds p s bounds := by
  have blocks : 0<p.blocks := Nat.zero_lt_of_lt c.isLt
  have hf : KernelOnly.first p s=first xs r h := by
    simpa [KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,first] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 0 blocks
  have hl : KernelOnly.last p s=last xs r h := by
    simpa [KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,last] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 1 blocks
  have slotIndex (slot : Int) (lo : KernelOnly.first p s≤slot) (hi : slot<KernelOnly.last p s) :
      ∃ v : Fin visits,v.val<count xs r h ∧ first xs r h+v.val=slot ∧
        csrColumn p s slot=column xs r h v := by
    have lower : first xs r h≤slot := hf ▸ lo
    have upper : slot<last xs r h := hl ▸ hi
    have live : (slot-first xs r h).toNat<count xs r h := by
      unfold count
      exact (Int.toNat_lt_toNat (lt_of_le_of_lt (sub_nonneg.mpr lower) (sub_lt_sub_right upper _))).mpr
        (sub_lt_sub_right upper _)
    have cap := (admitted r h blocks).2.2.2.1
    let v : Fin visits := ⟨(slot-first xs r h).toNat,lt_of_lt_of_le live cap⟩
    have equal : first xs r h+v.val=slot := by
      dsimp [v]
      rw [Int.toNat_of_nonneg (sub_nonneg.mpr lower)]
      rw [← Int.add_sub_assoc,Int.add_comm (first xs r h) slot,Int.add_sub_cancel]
    refine ⟨v,live,equal,?_⟩
    have read := column_binding a native encode observe p batches visits xs s binding r h v blocks live
    simpa only [csrColumn,KernelOnly.layout,KernelOnly.head,ids,PhysicalKernel.Program.pids,
      layout,head,← equal,Int.add_assoc] using read
  constructor
  · have hb := bounded.2.1 (0 : Fin 5) (r,h,0,PUnit.unit) blocks
    simpa [io,inputOffset,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      ids,PhysicalKernel.Program.pids,layout,head] using hb
  · have hb := bounded.2.1 (0 : Fin 5) (r,h,1,PUnit.unit) blocks
    simpa [io,inputOffset,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      ids,PhysicalKernel.Program.pids,layout,head] using hb
  · intro slot lo hi
    obtain ⟨v,live,equal,_⟩ := slotIndex slot lo hi
    have hb := bounded.2.1 (1 : Fin 5) (r,h,v,PUnit.unit) ⟨blocks,live⟩
    simpa only [io,inputOffset,inputs,KernelOnly.layout,KernelOnly.head,ids,
      PhysicalKernel.Program.pids,layout,head,← equal,Int.add_assoc] using hb
  · intro slot lo hi block blockLt i active
    obtain ⟨v,live,_,_⟩ := slotIndex slot lo hi
    have mask : inputActive p batches visits (2 : Fin 5) xs
        (r,h,v,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) := by
      exact ⟨live,by simpa only [KernelOnly.row,ids,PhysicalKernel.Program.pids] using active⟩
    have hb := bounded.2.1 (2 : Fin 5) (r,h,v,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) mask
    simpa [io,inputOffset,KernelOnly.qPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.row,
      ids,PhysicalKernel.Program.pids,batch,head,Nat.add_assoc] using hb
  · intro slot lo hi block blockLt i active
    obtain ⟨v,live,_,col⟩ := slotIndex slot lo hi
    have mask : inputActive p batches visits (3 : Fin 5) xs
        (r,h,v,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) := by
      exact ⟨live,by simpa only [col] using active⟩
    have hb := bounded.2.1 (3 : Fin 5) (r,h,v,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) mask
    simpa [io,inputOffset,KernelOnly.kPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.kv,
      ids,PhysicalKernel.Program.pids,batch,head,kv,col,Nat.add_assoc] using hb
  · intro slot lo hi i active
    obtain ⟨v,live,_,col⟩ := slotIndex slot lo hi
    have mask : inputActive p batches visits (4 : Fin 5) xs
        (r,h,c,v,i.1,i.2.1,PUnit.unit) := by
      exact ⟨live,by simpa only [col] using active⟩
    have hb := bounded.2.1 (4 : Fin 5) (r,h,c,v,i.1,i.2.1,PUnit.unit) mask
    simpa [io,inputOffset,KernelOnly.vPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.kv,KernelOnly.channel,
      ids,PhysicalKernel.Program.pids,batch,head,kv,col,Nat.add_assoc] using hb
  · intro i active
    have mask : (io a native encode observe p batches visits).outputMask (0 : Fin 1) xs
        (r,h,c,i.1,i.2.1,PUnit.unit) := by
      simpa only [KernelOnly.row,ids,PhysicalKernel.Program.pids] using active
    have hb := bounded.2.2.1 (0 : Fin 1) (r,h,c,i.1,i.2.1,PUnit.unit) mask
    simpa [io,outputOffset,KernelOnly.outputOffset,KernelOnly.batch,KernelOnly.head,KernelOnly.row,
      ids,PhysicalKernel.Program.pids,batch,head] using hb

#axiomsClean access_bounds
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
