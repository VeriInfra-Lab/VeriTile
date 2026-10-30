import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Physical

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
open VeriTile.Triton FP.Structural Contract
noncomputable section

/-- A sentinel segment may overlap another live segment. The correct frame is
for cells untouched by every active segment, and preserves all four bytes. -/
theorem untouched_cell_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State)
    (ok : Admitted t s spans (lists t before)) (cell : Nat×Nat)
    (hc : cell.2<(shape t s).outputColumns)
    (inactive : ¬Touched (shape t s) (segments t s (lists t before)) cell) (byte : Fin 4) :
    sourceStores rest ops t s before t.y.device.index ((key t s cell).address+byte.val)=
      before t.y.device.index ((key t s cell).address+byte.val) := by
  apply source_frame
  intro other live covered
  have inside:=touched_inside ok other live
  have hbyte:=byte.isLt
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,tensorKey] at lo hi
  have same : offset (shape t s) other=offset (shape t s) cell := by omega
  have coordinates:=linear_injective inside.2 hc same
  have equal : other=cell := Prod.ext coordinates.1 coordinates.2
  exact inactive (equal ▸ live)

theorem output_padding_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State)
    (ok : Admitted t s spans (lists t before)) (cell : Nat×Nat)
    (hc : cell.2<(shape t s).outputColumns)
    (outside : cell.2<(shape t s).slice ∨ (shape t s).slice+(shape t s).columns≤cell.2)
    (byte : Fin 4) :
    sourceStores rest ops t s before t.y.device.index ((key t s cell).address+byte.val)=
      before t.y.device.index ((key t s cell).address+byte.val) := by
  apply untouched_cell_frame rest ops t s before ok cell hc ?_ byte
  rintro ⟨segment,member,active,rowLo,rowHi,colLo,colHi⟩
  omega

#axiomsClean untouched_cell_frame
#axiomsClean output_padding_frame
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
