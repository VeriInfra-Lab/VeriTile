import bench.tritonbench_g.layer_norm_liger.Proofs.Grid
namespace VeriTile.Bench.Repairs.LayerNormLiger.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

theorem row_offset_ne (stride col other : Nat) (hc : col<stride) (ho : other<stride)
    (pid later : Nat) (hne : pid≠later) : pid*stride+col≠later*stride+other := by
  intro he
  exact hne (Launch.row_major_injective hc ho he).1

theorem backward_dw {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (base : State α) (programs pid : Nat) (hp : pid<programs)
    (hwd : p.dw≠p.db) (hxd : p.dw≠p.dx) (hs : p.columns≤p.dwStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (reference (Backward.reference M R p) base programs).mem p.dw (pid*p.dwStride+i.1.val)=
      Backward.stored M ((Backward.accumulated M R p
        (launch base (reference (Backward.reference M R p) base pid).mem pid)).dw i) := by
  rw [readback _ base programs pid hp p.dw (pid*p.dwStride+i.1.val) ?_]
  · exact Backward.finish_dw M p pid _ hwd i hi
  · intro later hl hu mem
    apply Backward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inr ?_,Or.inl hwd,Or.inl hxd⟩
    intro j hj
    exact row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs)
      pid later (by omega)

theorem backward_db {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (base : State α) (programs pid : Nat) (hp : pid<programs)
    (hbw : p.db≠p.dw) (hbx : p.db≠p.dx) (hs : p.columns≤p.dbStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (reference (Backward.reference M R p) base programs).mem p.db (pid*p.dbStride+i.1.val)=
      Backward.stored M ((Backward.accumulated M R p
        (launch base (reference (Backward.reference M R p) base pid).mem pid)).db i) := by
  rw [readback _ base programs pid hp p.db (pid*p.dbStride+i.1.val) ?_]
  · exact Backward.finish_db M p pid _ i hi
  · intro later hl hu mem
    apply Backward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inl hbw,Or.inr ?_,Or.inl hbx⟩
    intro j hj
    exact row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs)
      pid later (by omega)

/-- Every program's actual DX stores survive the rest of the grid. The local
loop value is the explicit complete scalar recurrence, not an output premise. -/
theorem backward_dx {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (base : State α) (programs pid row : Nat) (hp : pid<programs)
    (hr : row<min ((pid+1)*p.rowsPerProgram) p.rows)
    (hxw : p.dx≠p.dw) (hxb : p.dx≠p.db) (hs : p.columns≤p.dxStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (reference (Backward.reference M R p) base programs).mem p.dx (row*p.dxStride+i.1.val)=
      (Backward.accumulated M R p
        (launch base (reference (Backward.reference M R p) base pid).mem pid)).state.mem
          p.dx (row*p.dxStride+i.1.val) := by
  rw [readback _ base programs pid hp p.dx (row*p.dxStride+i.1.val) ?_]
  · exact Backward.finish_frame M p pid _ _ _ (Or.inl hxw) (Or.inl hxb)
  · intro later hl hu mem
    apply Backward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inl hxw,Or.inl hxb,Or.inr ?_⟩
    intro r hlo hhi j hj he
    have heq := (Launch.row_major_injective (Nat.lt_of_lt_of_le hi hs)
      (Nat.lt_of_lt_of_le hj hs) he).1
    have hm : (pid+1)*p.rowsPerProgram≤later*p.rowsPerProgram :=
      Nat.mul_le_mul_right _ (by omega)
    have hr' : row<(pid+1)*p.rowsPerProgram := lt_of_lt_of_le hr (Nat.min_le_left _ _)
    simp only [Backward.start,launch,if_true] at hlo
    omega

theorem backward_inputs {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (base : State α) (programs : Nat) (r : RegionName)
    (hw : r≠p.dw) (hb : r≠p.db) (hx : r≠p.dx) :
    (reference (Backward.reference M R p) base programs).mem r=base.mem r := by
  funext o
  apply frame
  intro pid hp mem
  exact Backward.reference_frame M R p _ _ _ ⟨Or.inl hw,Or.inl hb,Or.inl hx⟩

theorem forward_mean {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Forward.Parameters) (base : State α) (rows row : Nat) (hr : row<rows)
    (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hs : 0<p.meanStride) :
    (reference (Forward.reference M R p) base rows).mem p.mean (row*p.meanStride)=
      Forward.stored M (Forward.meanValue M R p
        (launch base (reference (Forward.reference M R p) base row).mem row)) := by
  rw [readback _ base rows row hr p.mean (row*p.meanStride) ?_]
  · exact Forward.reference_mean M R p _ hmy hmr
  · intro later hl hu mem
    apply Forward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inr ?_,Or.inl hmr,Or.inl hmy⟩
    simp only [Forward.meanOffset,launch,if_true]
    have hh := Nat.mul_lt_mul_of_pos_right hl hs
    omega

theorem forward_rstd {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Forward.Parameters) (base : State α) (rows row : Nat) (hr : row<rows)
    (hry : p.rstd≠p.y) (hrm : p.rstd≠p.mean) (hs : 0<p.rstdStride) :
    (reference (Forward.reference M R p) base rows).mem p.rstd (row*p.rstdStride)=
      Forward.stored M (Forward.rstdValue M R p
        (launch base (reference (Forward.reference M R p) base row).mem row)) := by
  rw [readback _ base rows row hr p.rstd (row*p.rstdStride) ?_]
  · exact Forward.reference_rstd M R p _ hry
  · intro later hl hu mem
    apply Forward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inl hrm,Or.inr ?_,Or.inl hry⟩
    simp only [Forward.rstdOffset,launch,if_true]
    have hh := Nat.mul_lt_mul_of_pos_right hl hs
    omega

theorem forward_y {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Forward.Parameters) (base : State α) (rows row : Nat) (hr : row<rows)
    (hym : p.y≠p.mean) (hyr : p.y≠p.rstd) (hs : p.columns≤p.yStride)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (reference (Forward.reference M R p) base rows).mem p.y (row*p.yStride+i.1.val)=
      Forward.stored M (Forward.yValue M R p
        (launch base (reference (Forward.reference M R p) base row).mem row) i) := by
  rw [readback _ base rows row hr p.y (row*p.yStride+i.1.val) ?_]
  · exact Forward.reference_y M R p _ i hi
  · intro later hl hu mem
    apply Forward.reference_frame M R p (launch base mem later)
    refine ⟨Or.inl hym,Or.inl hyr,Or.inr ?_⟩
    intro j hj
    exact row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs)
      row later (by omega)

theorem forward_inputs {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Forward.Parameters) (base : State α) (rows : Nat) (r : RegionName)
    (hm : r≠p.mean) (hr : r≠p.rstd) (hy : r≠p.y) :
    (reference (Forward.reference M R p) base rows).mem r=base.mem r := by
  funext o
  apply frame
  intro pid hp mem
  exact Forward.reference_frame M R p _ _ _ ⟨Or.inl hm,Or.inl hr,Or.inl hy⟩

#axiomsClean backward_dw
#axiomsClean backward_db
#axiomsClean backward_dx
#axiomsClean backward_inputs
#axiomsClean forward_mean
#axiomsClean forward_rstd
#axiomsClean forward_y
#axiomsClean forward_inputs
end
end VeriTile.Bench.Repairs.LayerNormLiger.Grid
