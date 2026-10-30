import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.InputBindings
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.FormulaCongruence

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

variable (a : Arithmetic β) (native : Bool) (zero : α) (encode : α→β) (observe : β→α)
  (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
  (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
include binding

theorem query_lookup_binding (j : TileIndex [ceil p.length 16,batches*p.heads,visits,p.blocks,16,16])
    (live : inputActive p batches visits (2 : Fin 5) xs j) :
    broadcast (encode (memory zero p batches visits xs .q (inputOffset p batches visits (2 : Fin 5) xs j)))=
      (s.mem "Q" (inputOffset p batches visits (2 : Fin 5) xs j)).read .real := by
  exact lookup_bound _ _ (xs 2) zero (fun x=>broadcast (encode x))
    (fun address=>(s.mem "Q" address).read .real)
    (fun i hi=>(binding (2 : Fin 5) i hi).symm) j live

theorem key_lookup_binding (j : TileIndex [ceil p.length 16,batches*p.heads,visits,p.blocks,16,16])
    (live : inputActive p batches visits (3 : Fin 5) xs j) :
    broadcast (encode (memory zero p batches visits xs .k (inputOffset p batches visits (3 : Fin 5) xs j)))=
      (s.mem "K" (inputOffset p batches visits (3 : Fin 5) xs j)).read .real := by
  exact lookup_bound _ _ (xs 3) zero (fun x=>broadcast (encode x))
    (fun address=>(s.mem "K" address).read .real)
    (fun i hi=>(binding (3 : Fin 5) i hi).symm) j live

theorem value_lookup_binding (j : TileIndex [ceil p.length 16,batches*p.heads,p.blocks,visits,16,16])
    (live : inputActive p batches visits (4 : Fin 5) xs j) :
    broadcast (encode (memory zero p batches visits xs .v (inputOffset p batches visits (4 : Fin 5) xs j)))=
      (s.mem "V" (inputOffset p batches visits (4 : Fin 5) xs j)).read .real := by
  exact lookup_bound _ _ (xs 4) zero (fun x=>broadcast (encode x))
    (fun address=>(s.mem "V" address).read .real)
    (fun i hi=>(binding (4 : Fin 5) i hi).symm) j live

theorem input_agreement (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) :
    InputAgreement a p s (reified zero encode p batches visits xs s.pids) := by
  let t := reified zero encode p batches visits xs s.pids
  have blocks : 0<p.blocks := Nat.zero_lt_of_lt c.isLt
  have hf : KernelOnly.first p s=first xs r h := by
    simpa [KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,first] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 0 blocks
  have hl : KernelOnly.last p s=last xs r h := by
    simpa [KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,last] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 1 blocks
  have hft : KernelOnly.first p t=KernelOnly.first p s := by
    simpa [t,reified,KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      ids,PhysicalKernel.Program.pids,layout,head] using
      rows_lookup_binding a native encode observe p batches visits xs s binding r h 0 blocks
  have hlt : KernelOnly.last p t=KernelOnly.last p s := by
    simpa [t,reified,KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      ids,PhysicalKernel.Program.pids,layout,head] using
      rows_lookup_binding a native encode observe p batches visits xs s binding r h 1 blocks
  have hc : visitCount p s=count xs r h := by simp only [visitCount,hf,hl,count]
  have finite (v : Nat) (hv : v<visitCount p s) : v<visits := by
    rw [hc] at hv
    exact lt_of_lt_of_le hv (admitted r h blocks).2.2.2.1
  have col (v : Nat) (hv : v<visitCount p s) :
      csrColumn p s (KernelOnly.first p s+v)=column xs r h ⟨v,finite v hv⟩ := by
    simpa only [csrColumn,hf,KernelOnly.layout,KernelOnly.head,ids,
      PhysicalKernel.Program.pids,layout,head,Int.add_assoc] using
      column_binding a native encode observe p batches visits xs s binding r h ⟨v,finite v hv⟩
        blocks (hc ▸ hv)
  have colt (v : Nat) (hv : v<visitCount p s) :
      csrColumn p t (KernelOnly.first p t+v)=csrColumn p s (KernelOnly.first p s+v) := by
    rw [hft,hf]
    simpa [csrColumn,t,reified,columns,KernelOnly.layout,KernelOnly.head,ids,
      PhysicalKernel.Program.pids,layout,head,Int.add_assoc] using
      columns_lookup_binding a native encode observe p batches visits xs s binding r h ⟨v,finite v hv⟩
        blocks (hc ▸ hv)
  refine ⟨rfl,hft.symm,hlt.symm,fun v hv=>(colt v hv).symm,?_,?_,?_⟩
  · intro v hv block blockLt
    funext i
    unfold KernelOnly.query
    have roweq : KernelOnly.row t i.1=KernelOnly.row s i.1 := rfl
    rw [roweq]
    split_ifs with live
    · have mask : inputActive p batches visits (2 : Fin 5) xs
          (r,h,⟨v,finite v hv⟩,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) := by
        exact ⟨hc ▸ hv,by simpa [KernelOnly.row,ids,PhysicalKernel.Program.pids] using live⟩
      have read := query_lookup_binding a native zero encode observe p batches visits xs s binding
        (r,h,⟨v,finite v hv⟩,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) mask
      simpa [inputOffset,t,reified,KernelOnly.qPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.row,
        ids,PhysicalKernel.Program.pids,batch,head,Nat.add_assoc] using read.symm
    · rfl
  · intro v hv block blockLt
    funext i
    unfold KernelOnly.key
    rw [colt v hv]
    split_ifs with live
    · have mask : inputActive p batches visits (3 : Fin 5) xs
          (r,h,⟨v,finite v hv⟩,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) := by
        exact ⟨hc ▸ hv,by simpa [col v hv] using live⟩
      have read := key_lookup_binding a native zero encode observe p batches visits xs s binding
        (r,h,⟨v,finite v hv⟩,⟨block,blockLt⟩,i.1,i.2.1,PUnit.unit) mask
      simpa [inputOffset,t,reified,KernelOnly.kPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.kv,
        ids,PhysicalKernel.Program.pids,batch,head,kv,col v hv,Nat.add_assoc] using read.symm
    · rfl
  · intro v hv
    funext i
    unfold KernelOnly.value
    rw [colt v hv]
    split_ifs with live
    · have mask : inputActive p batches visits (4 : Fin 5) xs
          (r,h,c,⟨v,finite v hv⟩,i.1,i.2.1,PUnit.unit) := by
        exact ⟨hc ▸ hv,by simpa [col v hv] using live⟩
      have read := value_lookup_binding a native zero encode observe p batches visits xs s binding
        (r,h,c,⟨v,finite v hv⟩,i.1,i.2.1,PUnit.unit) mask
      simpa [inputOffset,t,reified,KernelOnly.vPointer,KernelOnly.batch,KernelOnly.head,KernelOnly.kv,KernelOnly.channel,
        ids,PhysicalKernel.Program.pids,batch,head,kv,col v hv,Nat.add_assoc] using read.symm
    · rfl

#axiomsClean input_agreement
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
