import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedInputPost
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedStoredScalars

/-! Complete source row observation. Arithmetic and comparisons remain opaque;
the target is the ordered input-only formula, and every store is read back
from the actual typed source execution. This is not the native fused graph. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

theorem affineValue_reads {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (s initial : State α) (h : ReadsEqual f r s initial)
    (bias : Bool) (weight biasRegion : RegionName)
    (hw : Domain.readRegion f r weight) (hb : bias=Bool.true → Domain.readRegion f r biasRegion)
    (N col : Nat) (x : α) :
    affineValue A bias x (raw A s weight 0 N col) (raw A s biasRegion 0 N col)=
    affineValue A bias x (raw A initial weight 0 N col) (raw A initial biasRegion 0 N col) := by
  cases hf : bias
  · simp [affineValue,raw,hf,h weight hw]
  · simp [affineValue,raw,hf,h weight hw,h biasRegion (hb hf)]

def RowPost {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block row : Nat) (s t : State α) : Prop :=
  let v := combinedValues A cmp f s r stride M N p row block
  Frame f r stride M N row s t ∧ InputPost A cmp f r stride M N p block row s t ∧
  (∀ j : TileIndex [block], j.1.val<N →
    t.mem r.y (row*stride.y+j.1.val)=Cell.mk .real
      (affineValue A f.bias (normalizedValues A f.rms N eps v j)
        (raw A s r.weight 0 N j.1.val) (raw A s r.bias 0 N j.1.val))) ∧
  (f.w1=Bool.true → ∀ j : TileIndex [block], j.1.val<N →
    t.mem r.y1 (row*stride.y1+j.1.val)=Cell.mk .real
      (affineValue A f.b1 (normalizedValues A f.rms N eps v j)
        (raw A s r.weight1 0 N j.1.val) (raw A s r.bias1 0 N j.1.val))) ∧
  t.mem r.rstd row=Cell.mk .real (cast A (rstdValue A f.rms N eps v)) ∧
  (f.rms=Bool.false → t.mem r.mean row=Cell.mk .real (cast A (meanValue A N v)))

theorem source_row_post {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hgeo : NativeGeometry.Geometry f r stride M N)
    (hrow : s.pids 0<M) :
    ∃ t, FP.Structural.exec (engine A) (repaired f r stride M N eps p block hb) s=some t ∧
      RowPost A cmp f r stride M N eps p block (s.pids 0) s t := by
  have hn := hgeo.outputs
  simp only [List.nodup_cons,List.mem_cons,List.not_mem_nil,not_or,List.nodup_nil,and_true] at hn
  have h01 : r.y ≠ r.y1 := by simp_all [ne_comm]
  have h12 : r.rstd ≠ r.y := by simp_all [ne_comm]
  have h02 : r.rstd ≠ r.y1 := by simp_all [ne_comm]
  have h13 : r.mean ≠ r.y := by simp_all [ne_comm]
  have h03 : r.mean ≠ r.y1 := by simp_all [ne_comm]
  have h23 : r.mean ≠ r.rstd := by simp_all [ne_comm]
  have hkeepnames : ∀ region, region=r.residualOut ∨ region=r.dropoutMask →
      region ≠ r.y ∧ region ≠ r.y1 ∧ region ≠ r.rstd ∧ region ≠ r.mean := by
    intro region hr
    rcases hr with h|h <;> subst region <;> simp_all [ne_comm]
  obtain ⟨u,hu,hru,hfu,hvu,hpu⟩ := input_prefix_post A cmp f r stride M N eps p block hb s hsep hrow (by simp_all [ne_comm])
  obtain ⟨v,hv,hrv⟩ := normalization_success A f r stride M N eps p block hb (s.pids 0) u hru
  obtain ⟨t,ht⟩ := affine_success A f r stride M N eps p block hb (s.pids 0) v hrv
  have hfv := normalization_frame A f r stride M N eps p block hb (s.pids 0) u v hru hv
  have hft := affine_frame A f r stride M N eps p block hb (s.pids 0) v t hrv ht
  have hread := (hfu.trans hfv).readsEqual hsep
  obtain ⟨hx,hstd,hm⟩ := normalization_values A f r stride M N eps p block hb (s.pids 0) u v hru _ hvu hv
  have hkeep : ∀ region, region=r.residualOut ∨ region=r.dropoutMask →
      ∀ offset, t.mem region offset=u.mem region offset := by
    intro region hr offset
    rw [affine_preserves A f r stride M N eps p block hb (s.pids 0) v t hrv ht region offset
      (hkeepnames region hr).1
      (fun _ => (hkeepnames region hr).2.1)]
    exact normalization_preserves A f r stride M N eps p block hb (s.pids 0) u v hru hv region offset
      (hkeepnames region hr).2.2.1
      (fun _ => (hkeepnames region hr).2.2.2)
  have hresmem : t.mem r.residualOut=u.mem r.residualOut := funext (hkeep _ (Or.inl rfl))
  have hmaskmem : t.mem r.dropoutMask=u.mem r.dropoutMask := funext (hkeep _ (Or.inr rfl))
  refine ⟨t,?_,hfu.trans (hfv.trans hft),?_,?_,?_,?_,?_⟩
  · have hcode : (repaired f r stride M N eps p block hb).surfaceBody=
        (repaired f r stride M N eps p block hb).surfaceBody.take 15 ++
        normalization f r stride M N eps p block hb ++ affine f r stride M N eps p block hb := rfl
    change run (engine A) (repaired f r stride M N eps p block hb).surfaceBody s=some t
    rw [hcode]
    simp only [run_append,hu,hv,ht,Option.bind_some]
  · simpa only [InputPost,hresmem,hmaskmem] using hpu
  · intro j hj
    rw [affine_y_readback A f r stride M N eps p block hb (s.pids 0) v t hrv.1 _ hx hstd hm
      h01 ht j hj]
    rw [affineValue_reads A f r v s hread f.bias r.weight r.bias
      (by simp [Domain.readRegion]) (by intro hf; simp [Domain.readRegion,hf])]
  · intro hw1 j hj
    rw [affine_y1_readback A f r stride M N eps p block hb (s.pids 0) v t hrv.1 _ hx hstd hm hsep hw1 ht j hj]
    rw [affineValue_reads A f r v s hread f.b1 r.weight1 r.bias1
      (by simp [Domain.readRegion,hw1]) (by intro hf; simp [Domain.readRegion,hw1,hf])]
  · rw [affine_preserves A f r stride M N eps p block hb (s.pids 0) v t hrv ht _ _
      h12 (fun _ => h02)]
    exact normalization_rstd_readback A f r stride M N eps p block hb (s.pids 0) u v hru _ hvu hv
  · intro hrms
    rw [affine_preserves A f r stride M N eps p block hb (s.pids 0) v t hrv ht _ _
      h13 (fun _ => h03)]
    exact normalization_mean_readback A f r stride M N eps p block hb (s.pids 0) u v hru _ hvu hrms
      h23 hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
