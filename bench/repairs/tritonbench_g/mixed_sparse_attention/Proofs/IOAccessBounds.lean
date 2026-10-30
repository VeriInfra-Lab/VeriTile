import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOBinding
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOKernelSafety

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
open Classical
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem query_mem (c : Config) (xs : Values α c) (p : Program)
    (hp : p∈programs (grid c)) (i : TileIndex [c.BM,c.BD])
    (hi : row c p+i.1.val<sequence c xs p) :
    qbase c p+(row c p+i.1.val)*c.sqm+i.2.1.val*c.sqk∈reads c (0 : Fin 8) xs := by
  apply List.mem_flatMap.mpr
  refine ⟨p,hp,List.mem_map.mpr ⟨i,?_,rfl⟩⟩
  simp only [lanes,List.mem_filter,decide_eq_true_eq,hi,and_true]
  exact TileShape.mem_allIndices _ _

theorem sparse_mem (c : Config) (xs : Values α c) (p : Program) (ph : Bool)
    (n : Nat) (j : Fin c.BN) (d : Fin c.BD) (hj : keyActive c xs p ph n j)
    (ds ts : Nat) :
    kvbase c p+d.val*ds+selected c xs p ph n j*ts∈sparseReads c xs p ph ds ts := by
  apply List.mem_flatMap.mpr
  have hn : n<(if ph then columnCount c xs p else blockCount c xs p) := by
    cases ph <;> exact hj.2
  refine ⟨n,List.mem_range.mpr hn,List.mem_map.mpr ⟨(j,d,()),?_,rfl⟩⟩
  simp only [lanes,List.mem_filter,decide_eq_true_eq,hj,and_true]
  exact TileShape.mem_allIndices _ _

theorem key_mem (c : Config) (xs : Values α c) (p : Program) (hp : p∈programs (grid c))
    (ph : Bool) (n : Nat) (j : Fin c.BN) (d : Fin c.BD) (hj : keyActive c xs p ph n j) :
    kvbase c p+d.val*c.skk+selected c xs p ph n j*c.skn∈reads c (1 : Fin 8) xs := by
  apply List.mem_flatMap.mpr
  refine ⟨p,hp,?_⟩
  have h:=sparse_mem c xs p ph n j d hj c.skk c.skn
  cases ph
  · exact List.mem_append_left _ h
  · exact List.mem_append_right _ h

theorem value_mem (c : Config) (xs : Values α c) (p : Program) (hp : p∈programs (grid c))
    (ph : Bool) (n : Nat) (j : Fin c.BN) (d : Fin c.BD) (hj : keyActive c xs p ph n j) :
    kvbase c p+d.val*c.svk+selected c xs p ph n j*c.svn∈reads c (2 : Fin 8) xs := by
  apply List.mem_flatMap.mpr
  refine ⟨p,hp,?_⟩
  have h:=sparse_mem c xs p ph n j d hj c.svk c.svn
  cases ph
  · exact List.mem_append_left _ h
  · exact List.mem_append_right _ h

theorem actual_bounds (M : Algebra α) (D : Dot α) (c : Config) (layout : Contiguous c)
    (work : List Program) (xs : Values α c) (s : State α) (h : Reads c xs s)
    (hd : domain c xs) (capacity : RegionName → Nat) (hb : (io M D c work).InBounds capacity xs)
    (p : Program) (hp : p∈programs (grid c)) :
    IOSourceSafety.Bounds c (start (grid c) p s) capacity := by
  constructor
  · exact hb.2.1 (3 : Fin 8) _ (List.mem_map.mpr ⟨p,hp,rfl⟩)
  · exact hb.2.1 (4 : Fin 8) _ (List.mem_map.mpr ⟨p,hp,rfl⟩)
  · exact hb.2.1 (6 : Fin 8) _ (List.mem_map.mpr ⟨p,hp,rfl⟩)
  · intro n hn
    apply hb.2.1 (5 : Fin 8)
    apply List.mem_flatMap.mpr
    refine ⟨p,hp,List.mem_map.mpr ⟨n,?_,rfl⟩⟩
    apply List.mem_range.mpr
    simpa [Source.blockCount,Source.metadataIndex,start,ids,h.blocks,blockCount,slot] using hn
  · intro n j hj
    apply hb.2.1 (7 : Fin 8)
    apply List.mem_flatMap.mpr
    refine ⟨p,hp,List.mem_map.mpr ⟨n+j.val,?_,?_⟩⟩
    · exact List.mem_range.mpr ((KernelIO.Reads.keyActive c xs s h p Bool.true n j).mp hj).1
    · simp [slot,Source.metadataIndex,start,ids,Nat.add_assoc]
  · intro i hi
    have active : row c p+i.1.val<sequence c xs p := by
      simpa [Source.rowStart,Source.sequenceLength,start,ids,h.length,row,sequence] using hi
    simpa [Source.queryBase,Source.rowStart,start,ids,qbase,row] using
      hb.2.1 (0 : Fin 8) _ (query_mem c xs p hp i active)
  · intro ph n j d hj
    have hk:=hb.2.1 (1 : Fin 8) _ (key_mem c xs p hp ph n j d ((KernelIO.Reads.keyActive c xs s h p ph n j).mp hj))
    simpa [Source.keyValueBase,start,ids,kvbase,←KernelIO.Reads.selected c xs s h p ph n j] using hk
  · intro ph n j d hj
    have hv:=hb.2.1 (2 : Fin 8) _ (value_mem c xs p hp ph n j d ((KernelIO.Reads.keyActive c xs s h p ph n j).mp hj))
    simpa [Source.keyValueBase,start,ids,kvbase,←KernelIO.Reads.selected c xs s h p ph n j] using hv
  · intro i hi
    have hh:=((mem_programs (grid c) p.1 p.2.1 p.2.2).mp hp).2.1
    have hn:=active_row_bound c s (KernelIO.Reads.lengths c xs s h hd) p hp i hi
    let j : TileIndex [c.Z*c.H,c.T,c.BD] := (⟨p.2.1,hh⟩,⟨p.1*c.BM+i.1.val,hn⟩,i.2.1,())
    have live : active c xs j := by
      simpa [active,j,Source.rowStart,Source.sequenceLength,start,ids,h.length] using hi
    have bound:=hb.2.2.1 (0 : Fin 1) j live
    change Source.address c (start (grid c) p s) i<capacity c.out
    rw [source_address c layout]
    exact bound

#axiomsClean actual_bounds
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
