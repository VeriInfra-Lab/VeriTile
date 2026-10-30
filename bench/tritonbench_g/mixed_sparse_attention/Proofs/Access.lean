import bench.tritonbench_g.mixed_sparse_attention.Proofs.Host
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Access
open VeriTile Triton FP.Structural Source Whole
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

/-- Ordinary allocation extents, allowing shared read-only backing regions. -/
structure Storage (c : Config) (capacity : RegionName → Nat) : Prop where
  data : ∀ r∈[c.q,c.k,c.v,c.out],tensorSize c≤capacity r
  sequence : c.Z≤capacity (Region.cast c.seq)
  blocks : c.Z*c.H*c.NR≤capacity (Region.cast c.blocks)
  counts : c.Z*c.H*c.NR≤capacity (Region.cast c.counts)
  offsets : c.Z*c.H*c.NR*c.NS≤capacity (Region.cast c.offsets)
  columns : c.Z*c.H*c.NR*c.NV≤capacity (Region.cast c.columns)

theorem bases (c : Config) (layout : Contiguous c) (s : State α) :
    queryBase c s=s.pids 1*c.T*c.BD ∧ keyValueBase c s=s.pids 1*c.T*c.BD := by
  have hp : s.pids 1/c.H*c.H+s.pids 1%c.H=s.pids 1 := by
    simpa [Nat.mul_comm] using Nat.div_add_mod (s.pids 1) c.H
  constructor <;> simp only [queryBase,keyValueBase] <;>
    simp only [layout.qz,layout.qh,layout.kz,layout.kh] <;>
    calc
      _=(s.pids 1/c.H*c.H+s.pids 1%c.H)*c.T*c.BD := by ring
      _=_ := by rw [hp]

theorem query_address (c : Config) (layout : Contiguous c) (s : State α) (n d : Nat) :
    queryBase c s+n*c.sqm+d*c.sqk=logicalAddress c (s.pids 1) n d := by
  rw [(bases c layout s).1,layout.qm,layout.qk]
  simp only [logicalAddress]
  ring

theorem key_address (c : Config) (layout : Contiguous c) (s : State α) (n d : Nat) :
    keyValueBase c s+d*c.skk+n*c.skn=logicalAddress c (s.pids 1) n d := by
  rw [(bases c layout s).2,layout.kk,layout.kn]
  simp only [logicalAddress]
  ring

theorem value_address (c : Config) (layout : Contiguous c) (s : State α) (n d : Nat) :
    keyValueBase c s+d*c.svk+n*c.svn=logicalAddress c (s.pids 1) n d := by
  rw [(bases c layout s).2,layout.vk,layout.vn]
  simp only [logicalAddress]
  ring

theorem launched_coordinates (c : Config) (shape : Shape c) (p : Program)
    (hp : p∈programs (grid c)) : p.1<c.NR ∧ p.2.1<c.Z*c.H ∧ p.2.1/c.H<c.Z := by
  obtain ⟨hx,hy,_⟩:=(mem_programs (grid c) p.1 p.2.1 p.2.2).mp hp
  refine ⟨?_,hy,(Nat.div_lt_iff_lt_mul shape.heads).mpr hy⟩
  simpa [grid,←shape.rows] using hx

theorem key_bound (c : Config) (shape : Shape c) (s : State α) (metadata : Metadata c s)
    (p : Program) (hp : p∈programs (grid c)) (ph : Bool) (n : Nat) (j : Fin c.BN)
    (active : keyActive c (start (grid c) p s) ph n j) :
    columnValues c (start (grid c) p s) ph n j<c.T := by
  obtain ⟨hx,hy,hb⟩:=launched_coordinates c shape p hp
  have hl:=metadata.sequence _ hb
  cases ph
  · exact lt_of_lt_of_le active.1 hl
  · have hi:=metadata.columnIndex p.2.1 hy p.1 hx (n+j.val) active.1
    change n+j.val<columnCount c (start (grid c) p s) ∧
      n<columnCount c (start (grid c) p s) at active
    change (if n+j.val<columnCount c (start (grid c) p s) ∧
      n<columnCount c (start (grid c) p s) then _ else 0)<c.T
    rw [if_pos active]
    simpa [metadataIndex,start,ids,Nat.add_assoc] using lt_of_lt_of_le hi hl

/-- The exact masks and offsets of every source load/store family. -/
structure Safe (c : Config) (capacity : RegionName → Nat) (s : State α) : Prop where
  sequence : ∀ p∈programs (grid c),p.2.1/c.H<capacity (Region.cast c.seq)
  counts : ∀ p∈programs (grid c),
    metadataIndex c (start (grid c) p s)<capacity (Region.cast c.blocks) ∧
    metadataIndex c (start (grid c) p s)<capacity (Region.cast c.counts)
  offsets : ∀ p∈programs (grid c),∀ b,b<blockCount c (start (grid c) p s) →
    metadataIndex c (start (grid c) p s)*c.NS+b<capacity (Region.cast c.offsets)
  columns : ∀ p∈programs (grid c),∀ n,∀ j : Fin c.BN,
    keyActive c (start (grid c) p s) Bool.true n j →
    metadataIndex c (start (grid c) p s)*c.NV+n+j.val<capacity (Region.cast c.columns)
  query : ∀ p∈programs (grid c),∀ i : TileIndex [c.BM,c.BD],
    active c (start (grid c) p s) i →
    queryBase c (start (grid c) p s)+(rowStart c (start (grid c) p s)+i.1.val)*c.sqm+
      i.2.1.val*c.sqk<capacity c.q
  keyValue : ∀ p∈programs (grid c),∀ ph n,∀ j : Fin c.BN,∀ d<c.BD,
    keyActive c (start (grid c) p s) ph n j →
    keyValueBase c (start (grid c) p s)+d*c.skk+
      columnValues c (start (grid c) p s) ph n j*c.skn<capacity c.k ∧
    keyValueBase c (start (grid c) p s)+d*c.svk+
      columnValues c (start (grid c) p s) ph n j*c.svn<capacity c.v
  output : ∀ p∈programs (grid c),∀ i : TileIndex [c.BM,c.BD],
    active c (start (grid c) p s) i →address c (start (grid c) p s) i<capacity c.out

theorem safe (c : Config) (shape : Shape c) (capacity : RegionName → Nat)
    (storage : Storage c capacity) (s : State α) (metadata : Metadata c s) : Safe c capacity s := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · intro p hp
    exact lt_of_lt_of_le (launched_coordinates c shape p hp).2.2 storage.sequence
  · intro p hp
    obtain ⟨hx,hy,_⟩:=launched_coordinates c shape p hp
    have h:=metadata_bound (c.Z*c.H) c.NR p.2.1 p.1 hy hx
    exact ⟨lt_of_lt_of_le h storage.blocks,lt_of_lt_of_le h storage.counts⟩
  · intro p hp b hb
    obtain ⟨hx,hy,_⟩:=launched_coordinates c shape p hp
    exact lt_of_lt_of_le (selected_metadata_bound (c.Z*c.H) c.NR c.NS p.2.1 p.1 b _
      hy hx (metadata.blockCount _ hy _ hx) hb) storage.offsets
  · intro p hp n j hj
    obtain ⟨hx,hy,_⟩:=launched_coordinates c shape p hp
    have h:=selected_metadata_bound (c.Z*c.H) c.NR c.NV p.2.1 p.1 (n+j.val) _
      hy hx (metadata.columnCount _ hy _ hx) hj.1
    simpa only [metadataIndex,start,ids,ite_true,ite_false,Nat.add_assoc] using
      lt_of_lt_of_le h storage.columns
  · intro p hp i hi
    rw [query_address c shape.contiguous]
    have hh:=(launched_coordinates c shape p hp).2.1
    have hn:=active_row_bound c s metadata.sequence p hp i hi
    exact lt_of_lt_of_le (dense_bound (c.Z*c.H) c.T c.BD _ _ _ hh hn i.2.1.isLt)
      (storage.data c.q (by simp))
  · intro p hp ph n j d hd hj
    rw [key_address c shape.contiguous,value_address c shape.contiguous]
    have hh:=(launched_coordinates c shape p hp).2.1
    have hn:=key_bound c shape s metadata p hp ph n j hj
    have hb:=dense_bound (c.Z*c.H) c.T c.BD p.2.1 _ d hh hn hd
    exact ⟨lt_of_lt_of_le hb (storage.data c.k (by simp)),lt_of_lt_of_le hb (storage.data c.v (by simp))⟩
  · intro p hp i hi
    rw [source_address c shape.contiguous]
    have hh:=(launched_coordinates c shape p hp).2.1
    have hn:=active_row_bound c s metadata.sequence p hp i hi
    exact lt_of_lt_of_le (dense_bound (c.Z*c.H) c.T c.BD _ _ _ hh hn i.2.1.isLt)
      (storage.data c.out (by simp))

#axiomsClean safe
#specNonCircular Storage avoiding [FP.Structural.exec,Whole.invoke]
#specNonCircular Safe avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Access
