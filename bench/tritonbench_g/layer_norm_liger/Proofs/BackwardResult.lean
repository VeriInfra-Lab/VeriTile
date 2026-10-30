import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardContract
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem finish_db {α : Type} (M : Algebra α) (p : Parameters) (pid : Nat) (a : Acc α p)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (finish M p pid a).mem p.db (pid*p.dbStride+i.1.val) = stored M (a.db i) := by
  apply State.masked_scatter_readback
  · exact hi
  · intro x y hxy
    have hv : x.1.val=y.1.val := Nat.add_left_cancel hxy
    rcases x with ⟨x,⟨⟩⟩
    rcases y with ⟨y,⟨⟩⟩
    simpa [Fin.ext_iff] using hv

theorem finish_dw {α : Type} (M : Algebra α) (p : Parameters) (pid : Nat) (a : Acc α p)
    (hne : p.dw≠p.db) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (finish M p pid a).mem p.dw (pid*p.dwStride+i.1.val) = stored M (a.dw i) := by
  unfold finish
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hne)]
  apply State.masked_scatter_readback
  · exact hi
  · intro x y hxy
    have hv : x.1.val=y.1.val := Nat.add_left_cancel hxy
    rcases x with ⟨x,⟨⟩⟩
    rcases y with ⟨y,⟨⟩⟩
    simpa [Fin.ext_iff] using hv

theorem finish_frame {α : Type} (M : Algebra α) (p : Parameters) (pid : Nat) (a : Acc α p)
    (r : RegionName) (o : Nat)
    (hw : r≠p.dw ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o≠pid*p.dwStride+i.1.val)
    (hb : r≠p.db ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o≠pid*p.dbStride+i.1.val) :
    (finish M p pid a).mem r o = a.state.mem r o := by
  unfold finish
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ ?_,State.masked_scatter_frame _ _ _ _ _ _ _ ?_]
  · exact hw.imp_right (fun h i _ hi => h i hi)
  · exact hb.imp_right (fun h i _ hi => h i hi)

end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

def saved (p : Backward.Parameters) (s : State ℝ) : Backward.Acc ℝ p := ⟨s,fun _ => 0,fun _ => 0⟩
def groupDW (p : Backward.Parameters) (s : State ℝ) (i : Nat) :=
  ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s),
    bdy p (saved p s) row i*xhat (bx p (saved p s) row)
      (Backward.mean p (saved p s) row) (Backward.rstd p (saved p s) row) i
def groupDB (p : Backward.Parameters) (s : State ℝ) (i : Nat) :=
  ∑ row ∈ Finset.Ico (Backward.start p s) (Backward.stop p s), bdy p (saved p s) row i
def rowDX (p : Backward.Parameters) (s : State ℝ) (row i : Nat) :=
  dx p.columns (bx p (saved p s) row) (bw p (saved p s)) (bdy p (saved p s) row)
    (Backward.mean p (saved p s) row) (Backward.rstd p (saved p s) row) i

theorem reference_dw (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (fresh : FreshDX p) (hne : p.dw≠p.db) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference (algebra rest) sumPrimitive p s).mem p.dw (s.pids 0*p.dwStride+i.1.val) =
      .mk .fp32 (groupDW p s i.1.val) := by
  unfold Backward.reference
  rw [Backward.finish_dw _ _ _ _ hne i hi]
  change Cell.mk .fp32 ((Backward.accumulated (algebra rest) sumPrimitive p s).dw i) = _
  unfold Backward.accumulated
  rw [loop_dw rest p _ (saved p s) fresh (by intro r hr; rfl) _ _ i hi]
  simp [Backward.initial, Backward.zero, algebra, groupDW, saved]

theorem reference_db (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (fresh : FreshDX p) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference (algebra rest) sumPrimitive p s).mem p.db (s.pids 0*p.dbStride+i.1.val) =
      .mk .fp32 (groupDB p s i.1.val) := by
  unfold Backward.reference
  rw [Backward.finish_db _ _ _ _ i hi]
  change Cell.mk .fp32 ((Backward.accumulated (algebra rest) sumPrimitive p s).db i) = _
  unfold Backward.accumulated
  rw [loop_db rest p _ (saved p s) fresh (by intro r hr; rfl) _ _ i hi]
  simp [Backward.initial, Backward.zero, algebra, groupDB, saved]

theorem reference_dx (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (fresh : FreshDX p) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride)
    (row : Nat) (lo : Backward.start p s≤row) (hr : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference (algebra rest) sumPrimitive p s).mem p.dx (row*p.dxStride+i.1.val) =
      .mk .fp32 (rowDX p s row i.1.val) := by
  unfold Backward.reference
  rw [Backward.finish_frame _ _ _ _ _ _ (Or.inl hxd) (Or.inl hxb)]
  exact loop_dx rest p _ (saved p s) fresh (by intro r hr; rfl) hb hs _ _ row lo hr i hi

/-- All output values of the actual whole backward kernel, with no assumed
output equation or unproved iteration law. Freshness is allocator separation. -/
theorem backward_exec (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (fresh : FreshDX p) (hwd : p.dw≠p.db) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride) :
    ∃ t, FP.Structural.exec (withSum (algebra rest) sumPrimitive) (Backward.kernel p) s = some t ∧
      (∀ i : TileIndex [p.block], i.1.val<p.columns →
        t.mem p.dw (s.pids 0*p.dwStride+i.1.val)=.mk .fp32 (groupDW p s i.1.val) ∧
        t.mem p.db (s.pids 0*p.dbStride+i.1.val)=.mk .fp32 (groupDB p s i.1.val)) ∧
      (∀ row, Backward.start p s≤row → row<Backward.stop p s →
        ∀ i : TileIndex [p.block], i.1.val<p.columns →
          t.mem p.dx (row*p.dxStride+i.1.val)=.mk .fp32 (rowDX p s row i.1.val)) := by
  obtain ⟨t,ht,hm⟩ := Backward.complete_run (algebra rest) sumPrimitive p s
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    rw [hm]
    exact ⟨reference_dw rest p s fresh hwd i hi,reference_db rest p s fresh i hi⟩
  · intro row lo hr i hi
    rw [hm]
    exact reference_dx rest p s fresh hxd hxb hb hs row lo hr i hi

#axiomsClean reference_dw
#axiomsClean reference_db
#axiomsClean reference_dx
#axiomsClean backward_exec
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
