import bench.tritonbench_g.layer_norm_ops.Proofs.FPContract
import bench.tritonbench_g.layer_norm_ops.Proofs.FPSupplied
import bench.tritonbench_g.layer_norm_ops.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section

/-- Actual one-row RMS backward graph, including +0 accumulation of DW.
Input Rstd is supplied; no multi-row or host gradient aggregation is asserted. -/
specification fp32_rms_one_row_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hxd : p.dx≠p.dw) :
    ∃ t, FP.Structural.exec M (rmsKernel p) s=some t ∧ t.mem=(rowReference M p s Bool.false).mem ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=stored M (rowDX M p s Bool.false i) ∧
        t.mem p.dw (offset s p.columns i)=stored M (rowDW M p s Bool.false i)) ∧
      (∀ r o, RowOutside p s Bool.false r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := rms_row_run M p s
  refine ⟨t,he,hm,?_,?_⟩
  · intro i hi
    rw [hm]
    exact rms_row_values M p s hxd i hi
  · intro r o h
    rw [hm]
    exact row_frame M p s Bool.false r o h

/-- Actual plain-bias one-row DX/DW/DB, retaining both +0 accumulator adds.
Mean/Rstd are supplied statistics; there is no assumed exact real sum law. -/
specification fp32_plain_one_row_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db) (hdb : p.dw≠p.db) :
    ∃ t, FP.Structural.exec M (plainKernel p) s=some t ∧ t.mem=(rowReference M p s Bool.true).mem ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=stored M (rowDX M p s Bool.true i) ∧
        t.mem p.dw (offset s p.columns i)=stored M (rowDW M p s Bool.true i) ∧
        t.mem p.db (offset s p.columns i)=stored M (rowDB M p s i)) ∧
      (∀ r o, RowOutside p s Bool.true r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := plain_row_run M p s
  refine ⟨t,he,hm,?_,?_⟩
  · intro i hi
    rw [hm]
    exact plain_row_values M p s hxd hxb hdb i hi
  · intro r o h
    rw [hm]
    exact row_frame M p s Bool.true r o h

/-- Independent full-block C1 reduction stage with actual masks,
undef W tails, multiplication and division. -/
specification fp32_c1_reduction_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (c1Kernel p) s=some t ∧
      t.mem p.c1 (s.pids 0)=stored M (c1Value M p s) ∧
      (∀ r o, (r≠p.c1 ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := c1_run M p s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    exact State.write_other _ _ _ _ _ _ h

/-- Independent full-block C2 reduction stage with actual masks,
undef W tails, multiplication and division. -/
specification fp32_c2_reduction_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (c2Kernel p) s=some t ∧
      t.mem p.c2 (s.pids 0)=stored M (rowC2 M p s) ∧
      (∀ r o, (r≠p.c2 ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := c2_run M p s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    exact State.write_other _ _ _ _ _ _ h

/-- RMS/plain DX with supplied C1 and optional C2. Production of these inputs
is external to this independent stage, even though separate reduction contracts
are provided. -/
specification fp32_supplied_dx_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (plain : Bool) :
    ∃ t, FP.Structural.exec M (if plain then plainDXKernel p else rmsDXKernel p) s=some t ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=stored M (suppliedDX M p s plain i)) ∧
      (∀ r o, OutsideVector p s p.dx p.dxStride r o → t.mem r o=s.mem r o) := by
  have hr : ∃ t, FP.Structural.exec M (if plain then plainDXKernel p else rmsDXKernel p) s=some t ∧
      t.mem=(dxReference M p s plain).mem := by
    cases plain
    · exact rms_dx_run M p s
    · exact plain_dx_run M p s
  obtain ⟨t,he,hm⟩ := hr
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact scatter_value _ _ _ _ _ _ _ hi
  · intro r o h
    rw [hm]
    exact scatter_frame _ _ _ _ _ _ _ _ h

/-- Adds supplied DXBase and residual gradient, then writes both accepted
outputs; no claim is made that DXBase was produced by another stage. -/
specification fp32_residual_add_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (h : p.dresIn≠p.dx) :
    ∃ t, FP.Structural.exec M (residualKernel p) s=some t ∧
      (∀ i : TileIndex [p.block],i.1.val<p.columns →
        t.mem p.dx (offset s p.dxStride i)=stored M (residualValue M p s i) ∧
        t.mem p.dresIn (offset s p.dresInStride i)=stored M (residualValue M p s i)) ∧
      (∀ r o, (OutsideVector p s p.dx p.dxStride r o ∧
        OutsideVector p s p.dresIn p.dresInStride r o) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := residual_run M p s
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact residual_values M p s h i hi
  · intro r o h
    rw [hm]
    exact residual_frame M p s r o h

/-- The accepted supplied forward Y/residual-output store stages, also the
shared row-vector store shape. It preserves every active fp32 cell exactly;
it does not produce the supplied normalization result. -/
specification supplied_vector_store_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (stride N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (vectorStore .fp32 pre out stride N B) s=some t ∧
      (∀ i : TileIndex [B],i.1.val<N → t.mem out (offset s stride i)=
        .mk .fp32 ((s.mem pre (offset s stride i)).read .fp32)) ∧
      (∀ r o, (r≠out ∨ ∀ i : TileIndex [B],i.1.val<N → o≠offset s stride i) →
        t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := vector_store_run M pre out stride N B s
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact vector_value pre out stride N B s i hi
  · intro r o h
    rw [hm]
    exact vector_frame pre out stride N B s r o h

/-- Independent supplied mean scalar store, preserving the original stage boundary. -/
specification supplied_mean_store_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (meanStore .fp32 pre out) s=some t ∧
      t.mem out (s.pids 0)=.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32) ∧
      (∀ r o, (r≠out ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := mean_store_run M pre out s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    exact State.write_other _ _ _ _ _ _ h

/-- Independent supplied rstd scalar store, preserving the original stage boundary. -/
specification supplied_rstd_store_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (rstdStore .fp32 pre out) s=some t ∧
      t.mem out (s.pids 0)=.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32) ∧
      (∀ r o, (r≠out ∨ o≠s.pids 0) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := rstd_store_run M pre out s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    exact State.write_other _ _ _ _ _ _ h

#axiomsClean fp32_rms_one_row_correctness
#axiomsClean fp32_plain_one_row_correctness
#axiomsClean fp32_c1_reduction_correctness
#axiomsClean fp32_c2_reduction_correctness
#axiomsClean fp32_supplied_dx_correctness
#axiomsClean fp32_residual_add_correctness
#axiomsClean supplied_vector_store_correctness
#axiomsClean supplied_mean_store_correctness
#axiomsClean supplied_rstd_store_correctness
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPCorrect
