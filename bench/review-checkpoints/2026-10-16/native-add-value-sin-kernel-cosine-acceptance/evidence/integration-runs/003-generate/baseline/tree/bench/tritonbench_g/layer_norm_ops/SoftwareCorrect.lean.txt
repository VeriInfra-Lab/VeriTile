import bench.tritonbench_g.layer_norm_ops.Proofs.FPClosed
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.SoftwareCorrect
open VeriTile.Triton FP.Structural FPStages FPClosed
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-! Complete nine previously accepted stages at fp32 storage under the existing
+0-seeded increasing-lane reduction profile, separately rounded scalar ops,
and concrete natural-dimension encoding. No callbacks or final-output premise.
This does not bind the software tree, contraction or division to GPU code. -/

specification software_fp32_rms_one_row_correctness 
    (p : Parameters) (s : State Word) (hxd : p.dx≠p.dw) :
    ∃ t, FP.Structural.exec (backend p.columns) (rmsKernel p) s=some t ∧ t.mem=(bitRowReference p s Bool.false).mem ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=.mk .fp32 (bitDX p s Bool.false i) ∧
        t.mem p.dw (offset s p.columns i)=.mk .fp32 (bitDW p s Bool.false i)) ∧
      (∀ r o, RowOutside p s Bool.false r o → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s Bool.false
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_rms_one_row_correctness (backend p.columns) p s hxd

specification software_fp32_plain_one_row_correctness 
    (p : Parameters) (s : State Word) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db) (hdb : p.dw≠p.db) :
    ∃ t, FP.Structural.exec (backend p.columns) (plainKernel p) s=some t ∧ t.mem=(bitRowReference p s Bool.true).mem ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=.mk .fp32 (bitDX p s Bool.true i) ∧
        t.mem p.dw (offset s p.columns i)=.mk .fp32 (bitDW p s Bool.true i) ∧
        t.mem p.db (offset s p.columns i)=.mk .fp32 (bitDB p s i)) ∧
      (∀ r o, RowOutside p s Bool.true r o → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s Bool.true
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_plain_one_row_correctness (backend p.columns) p s hxd hxb hdb

specification software_fp32_c1_reduction_correctness 
    (p : Parameters) (s : State Word) :
    ∃ t, FP.Structural.exec (backend p.columns) (c1Kernel p) s=some t ∧
      t.mem p.c1 (s.pids 0)=.mk .fp32 (bitC1 p s) ∧
      (∀ r o, (r≠p.c1 ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s Bool.false
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_c1_reduction_correctness (backend p.columns) p s

specification software_fp32_c2_reduction_correctness 
    (p : Parameters) (s : State Word) :
    ∃ t, FP.Structural.exec (backend p.columns) (c2Kernel p) s=some t ∧
      t.mem p.c2 (s.pids 0)=.mk .fp32 (bitC2 p s) ∧
      (∀ r o, (r≠p.c2 ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s Bool.false
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_c2_reduction_correctness (backend p.columns) p s

specification software_fp32_supplied_dx_correctness 
    (p : Parameters) (s : State Word) (plain : Bool) :
    ∃ t, FP.Structural.exec (backend p.columns) (if plain then plainDXKernel p else rmsDXKernel p) s=some t ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=.mk .fp32 (bitSuppliedDX p s plain i)) ∧
      (∀ r o, OutsideVector p s p.dx p.dxStride r o → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s plain
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_supplied_dx_correctness (backend p.columns) p s plain

specification software_fp32_residual_add_correctness 
    (p : Parameters) (s : State Word) (h : p.dresIn≠p.dx) :
    ∃ t, FP.Structural.exec (backend p.columns) (residualKernel p) s=some t ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=.mk .fp32 (bitResidual p s i) ∧
        t.mem p.dresIn (offset s p.dresInStride i)=.mk .fp32 (bitResidual p s i)) ∧
      (∀ r o, (OutsideVector p s p.dx p.dxStride r o ∧
        OutsideVector p s p.dresIn p.dresInStride r o) → t.mem r o=s.mem r o) := by
  obtain ⟨hdx, hdw, hdbits, hc1, hc2, hsup, hres⟩ := values_closed unused p s Bool.false
  simpa only [backend, hdx, hdw, hdbits, hc1, hc2, hsup, hres, stored_closed, rowReference_closed] using
    FPCorrect.fp32_residual_add_correctness (backend p.columns) p s h

specification software_supplied_vector_store_correctness 
    (pre out : RegionName) (stride N B : Nat) (s : State Word) :
    ∃ t, FP.Structural.exec (backend N) (vectorStore .fp32 pre out stride N B) s=some t ∧
      (∀ i : TileIndex [B],i.1.val<N → t.mem out (offset s stride i)=
        .mk .fp32 ((s.mem pre (offset s stride i)).read .fp32)) ∧
      (∀ r o, (r≠out ∨ ∀ i : TileIndex [B],i.1.val<N → o≠offset s stride i) →
        t.mem r o=s.mem r o) := by
  exact FPCorrect.supplied_vector_store_correctness (backend N) pre out stride N B s

specification software_supplied_mean_store_correctness 
    (pre out : RegionName) (s : State Word) :
    ∃ t, FP.Structural.exec (backend 0) (meanStore .fp32 pre out) s=some t ∧
      t.mem out (s.pids 0)=.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32) ∧
      (∀ r o, (r≠out ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  exact FPCorrect.supplied_mean_store_correctness (backend 0) pre out s

specification software_supplied_rstd_store_correctness 
    (pre out : RegionName) (s : State Word) :
    ∃ t, FP.Structural.exec (backend 0) (rstdStore .fp32 pre out) s=some t ∧
      t.mem out (s.pids 0)=.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32) ∧
      (∀ r o, (r≠out ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  exact FPCorrect.supplied_rstd_store_correctness (backend 0) pre out s

#axiomsClean software_fp32_rms_one_row_correctness
#axiomsClean software_fp32_plain_one_row_correctness
#axiomsClean software_fp32_c1_reduction_correctness
#axiomsClean software_fp32_c2_reduction_correctness
#axiomsClean software_fp32_supplied_dx_correctness
#axiomsClean software_fp32_residual_add_correctness
#axiomsClean software_supplied_vector_store_correctness
#axiomsClean software_supplied_mean_store_correctness
#axiomsClean software_supplied_rstd_store_correctness
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.SoftwareCorrect
