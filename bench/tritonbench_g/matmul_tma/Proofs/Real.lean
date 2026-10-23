import bench.tritonbench_g.matmul_tma.Proofs.Layout
import bench.tritonbench_g.matmul_tma.Proofs.LegacyReal
import VeriTile.Triton.Memory.FlattenFrame

namespace VeriTile.Bench.TritonBenchG.MatmulTma.Real
open VeriTile.Triton FPValue Host
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def realKernel (p : Parameters) : ComputeKernel :=
  matmul_tma_f32_surface p.a p.b p.c p.rows p.columns p.inner p.aRow p.aInner
    p.bInner p.bColumn p.cRow p.cColumn p.blockRows p.blockColumns p.blockInner

/-- The real interpretation of the shared typed source erases floating casts,
for either constexpr branch; no floating equality is inferred. -/
theorem source_real_projection (p : Parameters) (half : Bool) :
    (source p half).eraseDType.toAlgKernel = (realKernel p).toAlgKernel := by
  cases half <;>
    simp [source, typed, outputDType, realKernel, matmul_tma_f32_surface,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
      ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
      VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals rfl

/-- Independent mathematical product of the logical views; every K term occurs
once. Neither the source nor its execution occurs in this definition. -/
def product (h : Shape) (a : TileIndex [h.rows,h.inner] → ℝ)
    (b : TileIndex [h.inner,h.columns] → ℝ) (i : TileIndex [h.rows,h.columns]) : ℝ :=
  ∑ k : Fin h.inner, a (i.1,k,PUnit.unit) * b (k,i.2.1,PUnit.unit)

def io (h : Shape) (f : Flags) (r : Regions) : MaskedTileShapedKernelIO₂ :=
  let p := parameters h f r
  matmulTmaF32IO p.a p.b p.c p.rows p.columns p.inner p.aRow p.aInner
    p.bInner p.bColumn p.cRow p.cColumn p.blockRows p.blockColumns p.blockInner

theorem io_correct (h : Shape) (f : Flags) (r : Regions) :
    (io h f r).Implements (fun _ _ a b i => product h a b i) := by
  exact matmul_tma_f32_io_correctness _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    (c_injective h f r)

/-- Whole matrix readback and frame for the same typed source's real projection.
Host bounds are discharged separately from the ordinary allocation contract. -/
theorem execution (h : Shape) (f : Flags) (r : Regions) (s : BlockState)
    (a : TileIndex [h.rows,h.inner] → ℝ) (b : TileIndex [h.inner,h.columns] → ℝ)
    (ha : ∀ i, s.readMem (parameters h f r).a (aOffset (parameters h f r) i) = a i)
    (hb : ∀ i, s.readMem (parameters h f r).b (bOffset (parameters h f r) i) = b i) :
    ∃ t, exec (source (parameters h f r) f.half).eraseDType.toAlgKernel s = some t ∧
      (∀ i, t.readMem (outputRegion r f) (i.1.val * h.columns + i.2.1.val) = product h a b i) ∧
      (∀ R off, R ≠ outputRegion r f ∨
        (∀ i : TileIndex [h.rows,h.columns], off ≠ i.1.val * h.columns + i.2.1.val) →
        t.mem R off = s.mem R off) := by
  rw [source_real_projection]
  obtain ⟨t,he,hv,hf⟩ := matmul_tma_f32_region_run
    (parameters h f r).a (parameters h f r).b (parameters h f r).c
    h.rows h.columns h.inner (parameters h f r).aRow (parameters h f r).aInner
    (parameters h f r).bInner (parameters h f r).bColumn
    (parameters h f r).cRow (parameters h f r).cColumn h.rows h.columns h.inner
    (c_injective h f r) s a b ha hb
  refine ⟨t,he,?_,?_⟩
  · intro i
    cases hh : f.half <;> simpa [parameters,product,matmulSpecOf,MatmulTma.cOffset,
      returnedTensor,firstOutput,contiguous,outputRegion,hh] using hv i
  · intro R off ho
    apply hf R off
    cases hh : f.half <;> simpa [parameters,MatmulTma.cOffset,returnedTensor,firstOutput,contiguous,
      outputRegion,hh] using ho

/-- The flat bridge also admits the discarded FP32 allocation and any other
framed allocation. Unlike the legacy IO convenience face, it does not require
that the allocation list contain exactly the three active regions. -/
theorem flat_execution (h : Shape) (f : Flags) (r : Regions) (s : BlockState)
    (a : TileIndex [h.rows,h.inner] → ℝ) (b : TileIndex [h.inner,h.columns] → ℝ)
    (ha : ∀ i, s.readMem (parameters h f r).a (aOffset (parameters h f r) i) = a i)
    (hb : ∀ i, s.readMem (parameters h f r).b (bOffset (parameters h f r) i) = b i)
    (storage : FlatAlloc) (binding : AllocationBinding h f r storage)
    (coverage : ∀ R, R ∉ storage.regions → storage.extent R = 0)
    (clean : s.undef = fun _ _ => 0) :
    ∃ t, exec (storage.flattenKernel (source (parameters h f r) f.half).eraseDType.toAlgKernel)
        (storage.flattenState s) = some t ∧
      (∀ i, t.readMem storage.flat (storage.addr (outputRegion r f)
        (i.1.val * h.columns + i.2.1.val)) = product h a b i) ∧
      (∀ R off, R ≠ storage.flat ∨
        (∀ i : TileIndex [h.rows,h.columns],
          off ≠ storage.addr (outputRegion r f) (i.1.val * h.columns + i.2.1.val)) →
        t.mem R off = (storage.flattenState s).mem R off) := by
  obtain ⟨t,he,hv,hf⟩ := execution h f r s a b ha hb
  obtain ⟨ba,bb,bc⟩ := all_accesses_bounded h f r storage binding
  have safe : (source (parameters h f r) f.half).eraseDType.toAlgKernel.TraceSafe storage.extent s := by
    rw [source_real_projection]
    exact matmul_tma_f32_traceSafe _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ba bb bc
  have covered : (source (parameters h f r) f.half).eraseDType.toAlgKernel.FlattenOk := by
    rw [source_real_projection]
    exact matmul_tma_f32_flattenOk _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  have outmem : outputRegion r f ∈ storage.regions := by
    cases hh : f.half
    · simpa [outputRegion,hh] using binding.first_mem
    · simpa [outputRegion,hh] using binding.replacement_mem hh
  refine ⟨storage.flattenState t, ?_, ?_, ?_⟩
  · rw [storage.exec_flatten binding.disjoint coverage _ s safe covered clean,he]
    rfl
  · intro i
    have bound : i.1.val * h.columns + i.2.1.val < storage.extent (outputRegion r f) :=
      lt_of_lt_of_le (rowMajor_bound i.1 i.2.1) (output_capacity h f r storage binding)
    rw [show (storage.flattenState t).readMem storage.flat
          (storage.addr (outputRegion r f) (i.1.val * h.columns + i.2.1.val)) =
          t.readMem (outputRegion r f) (i.1.val * h.columns + i.2.1.val) from
      storage.flattenState_readMem binding.disjoint t outmem bound]
    exact hv i
  · intro R off outside
    apply storage.flattenState_frame s t
      (fun R off => R = outputRegion r f ∧
        ∃ i : TileIndex [h.rows,h.columns], off = i.1.val * h.columns + i.2.1.val)
    · intro region offset hn
      apply hf region offset
      by_cases hr : region = outputRegion r f
      · right
        intro i hi
        exact hn ⟨hr,i,hi⟩
      · exact Or.inl hr
    · rcases outside with outside | outside
      · exact Or.inl outside
      · right
        rintro region offset ⟨rfl,i,rfl⟩
        exact outside i

#axiomsClean flat_execution
#axiomsClean source_real_projection
#axiomsClean io_correct
#axiomsClean execution
#specNonCircular product avoiding [source,exec,realKernel]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Real
