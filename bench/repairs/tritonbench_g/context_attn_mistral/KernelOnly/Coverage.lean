import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Footprint
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Mathematical

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelCoverage
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Checked Execution
noncomputable section
set_option maxHeartbeats 4000000

/-- Every active lane in the actual grid is an admitted physical coordinate. -/
def coordinate (c : Host.Call) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (on : c.active idx i) : Coord c :=
  ⟨(⟨(c.launchAt idx).batch,Host.batch_bound c idx⟩,
    ⟨(c.launchAt idx).head,Host.head_bound c idx⟩,
    ⟨Formula.queryRow c.layout (c.launchAt idx) i.1.val,by
      have hb := Host.block_bound c idx
      have hl : i.1.val<128 := c.blockM ▸ i.1.isLt
      simp only [Formula.queryRow,c.blockM]
      omega⟩,i.2.1),on⟩

theorem coordinate_offset (c : Host.Call) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (on : c.active idx i) :
    offset (coordinate c idx i on)=c.offset idx i := rfl

theorem coordinate_active (q : Coord c) :
    c.active (Mathematical.index q) (c.lane (row q) (dim q)) := by
  unfold Host.Call.active
  rw [Mathematical.index_launch]
  simpa [Formula.queryRow,Host.Call.lane,c.blockM,Nat.mul_comm,Nat.div_add_mod,row,batch] using q.property

theorem coordinate_locate (q : Coord c) :
    c.offset (Mathematical.index q) (c.lane (row q) (dim q))=offset q := by
  unfold Host.Call.offset
  rw [Mathematical.index_launch]
  simp [Formula.queryRow,Host.Call.lane,c.blockM,Nat.mul_comm,Nat.div_add_mod,offset]

/-- Exact equality of source-store coverage and physical-store coverage,
including gaps and intentionally underlaunched rows. -/
theorem writes_iff (c : Host.Call) (rg : RegionName) (off : Nat) :
    KernelFootprint.writes c (KernelReference.allPrograms c) rg off ↔
      ∃ q : Coord c,rg=c.regions.out ∧ off=offset q := by
  constructor
  · rintro ⟨idx,_,i,on,eq,addr⟩
    exact ⟨coordinate c idx i on,eq,addr⟩
  · rintro ⟨q,eq,addr⟩
    refine ⟨Mathematical.index q,KernelReference.allPrograms_complete _ _,c.lane (row q) (dim q),
      coordinate_active q,eq,?_⟩
    exact addr.trans (coordinate_locate q).symm

theorem coordinate_injective (c : Host.Call) (idx jdx : GridIndex c.grid)
    (i j : TileIndex [c.layout.BM,c.layout.D]) (hi : c.active idx i) (hj : c.active jdx j)
    (equal : coordinate c idx i hi=coordinate c jdx j hj) : idx=jdx ∧ i=j := by
  have hb := congrArg (fun q : Coord c => batch q) equal
  have hh := congrArg (fun q : Coord c => head q) equal
  have hr := congrArg (fun q : Coord c => row q) equal
  have hd := congrArg (fun q : Coord c => dim q) equal
  change (c.launchAt idx).batch=(c.launchAt jdx).batch at hb
  change (c.launchAt idx).head=(c.launchAt jdx).head at hh
  change Formula.queryRow c.layout (c.launchAt idx) i.1.val=Formula.queryRow c.layout (c.launchAt jdx) j.1.val at hr
  change i.2.1=j.2.1 at hd
  have il : i.1.val<128 := c.blockM ▸ i.1.isLt
  have jl : j.1.val<128 := c.blockM ▸ j.1.isLt
  simp only [Formula.queryRow,c.blockM] at hr
  have block : (c.launchAt idx).block=(c.launchAt jdx).block := by omega
  have lane : i.1=j.1 := Fin.ext (by omega)
  refine ⟨Host.launchAt_injective c hb hh block,?_⟩
  exact Prod.ext lane (Prod.ext hd (Subsingleton.elim _ _))

theorem source_writer_unique (g : Geometry c tokens) (idx jdx : GridIndex c.grid)
    (i j : TileIndex [c.layout.BM,c.layout.D]) (hi : c.active idx i) (hj : c.active jdx j)
    (equal : c.offset idx i=c.offset jdx j) : idx=jdx ∧ i=j := by
  exact coordinate_injective c idx jdx i j hi hj
    (offset_injective g (show offset (coordinate c idx i hi)=offset (coordinate c jdx j hj) from equal))

/-- Empty grids need no launch-only tensor or allocation admission. -/
theorem empty_programs (c : Host.Call) (empty : c.batches=0 ∨ c.heads=0 ∨ c.blocks=0) :
    KernelReference.allPrograms c=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro idx _
  have b := Host.batch_bound c idx
  have h := Host.head_bound c idx
  have k := Host.block_bound c idx
  rcases empty with eb|eh|ek <;> omega

theorem empty_coordinates (c : Host.Call) (empty : c.batches=0 ∨ c.heads=0 ∨ c.blocks=0)
    (q : Coord c) : False := by
  have b := q.val.1.isLt
  have h := q.val.2.1.isLt
  have r := q.val.2.2.1.isLt
  rcases empty with eb|eh|ek <;> simp_all

#axiomsClean writes_iff
#axiomsClean source_writer_unique
#axiomsClean empty_programs
#axiomsClean empty_coordinates
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelCoverage
