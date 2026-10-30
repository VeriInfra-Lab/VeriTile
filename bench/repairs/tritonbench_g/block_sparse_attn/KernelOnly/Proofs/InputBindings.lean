import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Spec
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.LocalExecution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

theorem lookup_bound {ι α β : Type} (active : ι→Prop) (offset : ι→Nat) (value : ι→α)
    (fallback : α) (encode : α→β) (read : Nat→β)
    (bound : ∀ j,active j → encode (value j)=read (offset j))
    (j : ι) (live : active j) :
    encode (lookup active offset value fallback (offset j))=read (offset j) := by
  classical
  unfold lookup
  split_ifs with found
  · obtain ⟨valid,equal⟩ := Classical.choose_spec found
    rw [bound _ valid,equal]
  · exact False.elim (found ⟨j,live,rfl⟩)

/-- Pure input-array reification, used only to apply existing independent
formula lemmas. It is not an alternative source execution. -/
def reified {α β : Type} [Inhabited β] (zero : α) (encode : α→β)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (ids : Nat→Nat) : State (Replicas β) where
  mem := fun region address =>
    if region="Q" then .mk .real (broadcast (encode (memory zero p batches visits xs .q address)))
    else if region="K" then .mk .real (broadcast (encode (memory zero p batches visits xs .k address)))
    else if region="V" then .mk .real (broadcast (encode (memory zero p batches visits xs .v address)))
    else if region="rows" then .mk .int (rows p batches visits xs address)
    else if region="cols" then .mk .int (columns p batches visits xs address)
    else .mk .real (broadcast (encode zero))
  regs := fun _ _ _=>none
  pids := ids
  numPids := fun _=>0
  undef := fun dtype _ _=>defaultValue dtype

theorem reified_views {α β : Type} [Inhabited β] (zero : α) (encode : α→β)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits) (ids : Nat→Nat) :
    Views (reified zero encode p batches visits xs ids)
      (fun region address=>encode (memory zero p batches visits xs region address))
      (rows p batches visits xs) (columns p batches visits xs) := by
  constructor <;> intro address <;> simp [reified]

variable {α β : Type} [Inhabited β]
variable (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
  (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
  (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
include binding

theorem endpoint_binding (r : Row p) (h : Head p batches) (e : Fin 2) (blocks : 0<p.blocks) :
    (s.mem "rows" (layout p h.val*p.rowStride+r.val+e.val)).read .int=
      xs 0 (r,h,e,PUnit.unit) := by
  exact binding (0 : Fin 5) (r,h,e,PUnit.unit) blocks

theorem column_binding (r : Row p) (h : Head p batches) (v : Fin visits)
    (blocks : 0<p.blocks) (live : v.val<count xs r h) :
    (s.mem "cols" ((layout p h.val*p.colStride : Nat)+first xs r h+v.val : Int).toNat).read .int=
      column xs r h v := by
  exact binding (1 : Fin 5) (r,h,v,PUnit.unit) ⟨blocks,live⟩

theorem rows_lookup_binding (r : Row p) (h : Head p batches) (e : Fin 2) (blocks : 0<p.blocks) :
    rows p batches visits xs (layout p h.val*p.rowStride+r.val+e.val)=
      (s.mem "rows" (layout p h.val*p.rowStride+r.val+e.val)).read .int := by
  change lookup _ _ _ _ _=_
  exact lookup_bound _ _ (xs 0) (0 : Int) id (fun address=>(s.mem "rows" address).read .int)
    (fun j hj=>(binding (0 : Fin 5) j hj).symm) (r,h,e,PUnit.unit) blocks

theorem columns_lookup_binding (r : Row p) (h : Head p batches) (v : Fin visits)
    (blocks : 0<p.blocks) (live : v.val<count xs r h) :
    columns p batches visits xs
        ((layout p h.val*p.colStride : Nat)+first xs r h+v.val : Int)=
      (s.mem "cols" ((layout p h.val*p.colStride : Nat)+first xs r h+v.val : Int).toNat).read .int := by
  exact lookup_bound _ _ (xs 1) (0 : Int) id (fun address=>(s.mem "cols" address).read .int)
    (fun j hj=>(binding (1 : Fin 5) j hj).symm) (r,h,v,PUnit.unit) ⟨blocks,live⟩

theorem metadata_reads (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) : ReadInt32 p s := by
  have blocks : 0<p.blocks := Nat.zero_lt_of_lt c.isLt
  have hf : KernelOnly.first p s=first xs r h := by
    simpa [KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,first] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 0 blocks
  have hl : KernelOnly.last p s=last xs r h := by
    simpa [KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,last] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 1 blocks
  have hc : visitCount p s=count xs r h := by simp only [visitCount,hf,hl,count]
  have domain := admitted r h blocks
  refine ⟨?_,?_⟩
  · exact ⟨hf.symm ▸ domain.1,hl.symm ▸ domain.2.1⟩
  · intro v hv
    rw [hc] at hv
    have limit : v<visits := lt_of_lt_of_le hv domain.2.2.2.1
    have read := column_binding a native encode observe p batches visits xs s binding r h ⟨v,limit⟩ blocks hv
    dsimp only [ColumnInt32,csrColumn]
    simp only [hf,KernelOnly.layout,KernelOnly.head,ids,PhysicalKernel.Program.pids]
    have result := (domain.2.2.2.2 ⟨v,limit⟩ hv).1
    rw [← read] at result
    simpa only [layout,head,Int.add_assoc] using result

#axiomsClean reified_views
#axiomsClean metadata_reads
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
