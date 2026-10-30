import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Kernels
namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry
abbrev Program := VeriTile.Bench.Repairs.BgmvExpandSlice.Geometry.Program
def outputOffset (h : Call) (batch row n : Nat) := batch*h.columns+row+h.slice+n
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry
open VeriTile.Triton

theorem output_injective (h : Call) (p q : Program h)
    (same : outputOffset h p.batch p.row 0=outputOffset h q.batch q.row 0) :
    p.batch=q.batch ∧ p.row=q.row := by
  have bound:=h.sliceInside
  have pr:=p.rowInside
  have qr:=q.rowInside
  have pos : 0<h.columns := by omega
  have rp : p.row+h.slice<h.columns := by omega
  have rq : q.row+h.slice<h.columns := by omega
  unfold outputOffset at same
  have both:=congrArg (fun v => v/h.columns) same
  have ep : (p.batch*h.columns+p.row+h.slice+0)/h.columns=p.batch := by
    rw [Nat.add_zero,Nat.add_assoc,Nat.add_comm (p.batch*h.columns),Nat.add_mul_div_right _ _ pos,Nat.div_eq_of_lt rp]
    simp
  have eq : (q.batch*h.columns+q.row+h.slice+0)/h.columns=q.batch := by
    rw [Nat.add_zero,Nat.add_assoc,Nat.add_comm (q.batch*h.columns),Nat.add_mul_div_right _ _ pos,Nat.div_eq_of_lt rq]
    simp
  dsimp only at both
  rw [ep,eq] at both
  exact ⟨both,by simpa [both] using same⟩

def ofCoord (h : Call) (c : Fin h.batches × Fin h.N) : Program h := ⟨c.1,c.2,c.1.isLt,c.2.isLt⟩
def programs (h : Call) : List (Program h) :=
  ((List.finRange h.batches).product (List.finRange h.N)).map (ofCoord h)

theorem programs_unique (h : Call) : (programs h).Pairwise
    (fun p q => outputOffset h p.batch p.row 0≠outputOffset h q.batch q.row 0) := by
  rw [programs,List.pairwise_map]
  apply ((List.nodup_finRange h.batches).product (List.nodup_finRange h.N)).imp
  intro a b different same
  apply different
  have eq:=output_injective h (ofCoord h a) (ofCoord h b) same
  exact Prod.ext (Fin.ext eq.1) (Fin.ext eq.2)

theorem programs_complete (h : Call) (p : Program h) : p∈programs h := by
  apply List.mem_map.mpr
  exact ⟨(⟨p.batch,p.batchInside⟩,⟨p.row,p.rowInside⟩),by simp,by cases p;rfl⟩

theorem active_output (h : Call) (p : Program h) :
    outputOffset h p.batch p.row 0<h.batches*h.columns := by
  have hb:=p.batchInside
  have hn:=p.rowInside
  have bound:=h.sliceInside
  have cm:=Nat.mul_le_mul_right h.columns (Nat.succ_le_of_lt hb)
  unfold outputOffset
  rw [Nat.succ_mul] at cm
  omega

#axiomsClean programs_unique
#axiomsClean programs_complete
#axiomsClean active_output
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry
