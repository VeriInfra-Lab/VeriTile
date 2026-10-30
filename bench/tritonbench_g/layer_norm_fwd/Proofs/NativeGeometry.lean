import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_WholeGrid

/-! Actual active row geometry. Optional output strides are irrelevant when
the corresponding store is disabled; empty or singleton row axes require no
artificial lower bound on an unused row step. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeGeometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000

def RowSeparated (step M N : Nat) : Prop :=
  ∀ row other, row<M → other<M → ∀ offset,
    Contract.inRow (row*step) N offset → Contract.inRow (other*step) N offset → row=other

theorem rowSeparated_of_step (step M N : Nat) (h : N≤step) : RowSeparated step M N :=
  fun _ _ _ _ _ ha hb => WholeGrid.inRow_unique h ha hb

theorem rowSeparated_singleton (step M N : Nat) (h : M≤1) : RowSeparated step M N := by
  intro row other hr ho _ _ _
  omega

structure Geometry (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) : Prop where
  outputs : [r.y1,r.y,r.rstd,r.mean,r.residualOut,r.dropoutMask].Nodup
  y : RowSeparated stride.y M N
  y1 : f.w1=Bool.true → RowSeparated stride.y1 M N
  residualOut : f.residualOut=Bool.true → RowSeparated stride.residualOut M N

theorem of_legacy (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (h : WholeGrid.Geometry r stride N) : Geometry f r stride M N :=
  ⟨h.outputs,rowSeparated_of_step _ _ _ h.y,
    fun _ => rowSeparated_of_step _ _ _ h.y1,
    fun _ => rowSeparated_of_step _ _ _ h.residualOut⟩

theorem written_row_unique (f : Flags) (r : Regions) (stride : Strides)
    (M N row other : Nat) (hgeo : Geometry f r stride M N)
    (hr : row<M) (ho : other<M) (region : RegionName) (offset : Nat)
    (ha : Contract.written f r stride M N row region offset)
    (hb : Contract.written f r stride M N other region offset) : row=other := by
  have hn := hgeo.outputs
  simp only [List.nodup_cons,List.mem_cons,not_or,List.nodup_nil,and_true] at hn
  rcases ha with ⟨hfa,ha,hia⟩ | ⟨ha,hia⟩ | ⟨ha,hia⟩ | ⟨hfa,ha,hia⟩ |
    ⟨hfa,ha,hia⟩ | ⟨hfa,hga,hxa,ha,hia⟩ | ⟨hfa,hga,ha,hia⟩ <;>
    rcases hb with ⟨hfb,hb,hib⟩ | ⟨hb,hib⟩ | ⟨hb,hib⟩ | ⟨hfb,hb,hib⟩ |
      ⟨hfb,hb,hib⟩ | ⟨hfb,hgb,hxb,hb,hib⟩ | ⟨hfb,hgb,hb,hib⟩
  all_goals try simp_all
  all_goals first
    | exact hgeo.y row other hr ho offset hia hib
    | exact hgeo.y1 hfb row other hr ho offset hia hib
    | exact hgeo.residualOut hfb row other hr ho offset hia hib
    | exact WholeGrid.inRow_unique (Nat.le_refl N) hia hib
    | (have h := WholeGrid.inRow_unique (Nat.le_refl N) hia hib; omega)

#axiomsClean rowSeparated_of_step
#axiomsClean rowSeparated_singleton
#axiomsClean of_legacy
#axiomsClean written_row_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeGeometry
