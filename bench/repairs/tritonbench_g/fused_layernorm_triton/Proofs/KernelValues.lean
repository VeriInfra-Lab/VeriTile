import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Native
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeResult
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.KernelValues
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

def observeCell (I : Native.Instructions) (P : Native.Profile) :=
  FP.Structural.mapCell (Native.observe I P)

def mean (xs : Nat → Nat → ℝ) (D row : Nat) : ℝ :=
  ((List.range D).map (xs row)).sum / D

def variance (xs : Nat → Nat → ℝ) (D row : Nat) : ℝ :=
  ((List.range D).map (fun col => (xs row col - mean xs D row)^2)).sum / D

def inverse (xs : Nat → Nat → ℝ) (D row : Nat) : ℝ :=
  1 / Real.sqrt (variance xs D row + (1e-05 : ℝ))

def output (xs : Nat → Nat → ℝ) (ws bs : Nat → ℝ) (D row col : Nat) : ℝ :=
  (xs row col - mean xs D row)*inverse xs D row*ws col+bs col

theorem real (rest : Algebra ℝ) (s : State ℝ)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (positive : 0<D)
    (xs : Nat → Nat → ℝ) (ws bs : Nat → ℝ)
    (hx : ∀ row,row<rows → ∀ col,col<D → (s.mem X (col+D*row)).read .real=xs row col)
    (hw : ∀ col,col<D → (s.mem W col).read .real=ws col)
    (hb : ∀ col,col<D → (s.mem Bias col).read .real=bs col)
    (sep : Separation X W Bias Y Mean Rstd) (c : Contract.Configuration) (complete : c.tree.Complete) :
    ∃ final,KernelProof.launch (ideal rest) realEqual X W Bias Y Mean Rstd rows D c s=some final ∧
      (∀ row,row<rows → final.mem Mean row=.mk .real (mean xs D row)) ∧
      (∀ row,row<rows → final.mem Rstd row=.mk .real (inverse xs D row)) ∧
      (∀ row,row<rows → ∀ col,col<D → final.mem Y (col+D*row)=.mk .bf16 (output xs ws bs D row col)) ∧
      (∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off) := by
  obtain ⟨final,run,effect⟩ := KernelProof.succeeds (ideal rest) realEqual X W Bias Y Mean Rstd rows D c s sep
  have means (row : Nat) (hr : row<rows) : final.mem Mean row=.mk .real (logicalMean s X D row) := by
    rw [effect.mean row (List.mem_range.mpr hr),
      (row_real_mean_m2 rest s X D c.block row positive c.positive _ (KernelProof.complete _ complete)).1]
  have inverses (row : Nat) (hr : row<rows) : final.mem Rstd row=.mk .real (logicalInverse s X D row) := by
    rw [effect.rstd row (List.mem_range.mpr hr),
      row_real_inverse rest s X D c.block row positive c.positive _ (KernelProof.complete _ complete)]
  have outputs (row : Nat) (hr : row<rows) (col : Nat) (hc : col<D) :
      final.mem Y (col+D*row)=.mk .bf16 (logicalOutput s X W Bias D row col) := by
    rw [effect.output row (List.mem_range.mpr hr) col hc,
      row_real_output rest s X W Bias D c.block row col positive c.positive _ (KernelProof.complete _ complete)]
  have frame : ∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off :=
    fun q off miss => effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))
  have logical (row : Nat) (hr : row<rows) : logicalRow s X D row=(List.range D).map (xs row) := by
    unfold logicalRow
    apply List.map_congr_left
    intro col hc
    exact hx row hr col (List.mem_range.mp hc)
  have avg (row : Nat) (hr : row<rows) : logicalMean s X D row=mean xs D row := by
    simp [logicalMean,logical row hr,mean]
  have inv (row : Nat) (hr : row<rows) : logicalInverse s X D row=inverse xs D row := by
    simp [logicalInverse,logicalVariance,logical row hr,avg row hr,inverse,variance,List.map_map,Function.comp_def]
  refine ⟨final,run,?_,?_,?_,frame⟩
  · intro row hr
    rw [means row hr,avg row hr]
  · intro row hr
    rw [inverses row hr,inv row hr]
  · intro row hr col hc
    rw [outputs row hr col hc]
    simp only [logicalOutput,hx row hr col hc,hw col hc,hb col hc,avg row hr,inv row hr,output]

theorem native (I : Native.Instructions) (P : Native.Profile) (s : State Native.Term)
    (X W B Y Mean Rstd : RegionName) (rows D : Nat)
    (xs : Nat → Nat → Native.Term) (ws bs : Nat → Native.Term)
    (hx : ∀ row,row<rows → ∀ col,col<D → (s.mem X (col+D*row)).read .real=xs row col)
    (hw : ∀ col,col<D → (s.mem W col).read .real=ws col)
    (hb : ∀ col,col<D → (s.mem B col).read .real=bs col)
    (sep : Separation X W B Y Mean Rstd) (c : Contract.Configuration) :
    ∃ final,KernelProof.launch Native.algebra (Native.equal I P) X W B Y Mean Rstd rows D c s=some final ∧
      (∀ row,row<rows →
        let ref := Native.reference I P (xs row) ws bs D c.block c.tree
        observeCell I P (final.mem Mean row)=.mk .real ref.mean ∧
        observeCell I P (final.mem Rstd row)=.mk .real ref.rstd ∧
        ∀ col,col<D → observeCell I P (final.mem Y (col+D*row))=.mk .bf16 (ref.output col)) ∧
      (∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off) := by
  obtain ⟨final,run,effect⟩ := KernelProof.succeeds Native.algebra (Native.equal I P)
    X W B Y Mean Rstd rows D c s sep
  refine ⟨final,run,?_,?_⟩
  · intro row hr
    have stats := Native.statistics_eq Native.algebra (Native.equal I P) s X D c.block row c.tree _ (hx row hr)
    refine ⟨?_,?_,?_⟩
    · rw [effect.mean row (List.mem_range.mpr hr),stats]
      rfl
    · rw [effect.rstd row (List.mem_range.mpr hr),stats]
      rfl
    · intro col hc
      rw [effect.output row (List.mem_range.mpr hr) col hc]
      unfold rowOutput
      rw [stats]
      simp only [normalizeValue,hx row hr col hc,hw col hc,hb col hc]
      rfl
  · intro q off miss
    exact effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))

theorem nativeSource (hoist : Bool) (I : Native.Instructions) (P : Native.Profile) (s : State Native.Term)
    (X W B Y Mean Rstd : RegionName) (rows D : Nat)
    (xs : Nat → Nat → Native.Term) (ws bs : Nat → Native.Term)
    (hx : ∀ row,row<rows → ∀ col,col<D → (s.mem X (col+D*row)).read .real=xs row col)
    (hw : ∀ col,col<D → (s.mem W col).read .real=ws col)
    (hb : ∀ col,col<D → (s.mem B col).read .real=bs col)
    (sep : Separation X W B Y Mean Rstd) (c : Contract.Configuration) :
    ∃ final,KernelProof.launchSource hoist Native.algebra (Native.equal I P) X W B Y Mean Rstd rows D c s=some final ∧
      (∀ row,row<rows →
        let ref := Native.reference I P (xs row) ws bs D c.block c.tree
        observeCell I P (final.mem Mean row)=.mk .real ref.mean ∧
        observeCell I P (final.mem Rstd row)=.mk .real ref.rstd ∧
        ∀ col,col<D → observeCell I P (final.mem Y (col+D*row))=.mk .bf16 (ref.output col)) ∧
      (∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off) := by
  obtain ⟨final,run,effect⟩ := KernelProof.succeedsSource hoist Native.algebra (Native.equal I P)
    X W B Y Mean Rstd rows D c s sep
  refine ⟨final,run,?_,?_⟩
  · intro row hr
    have stats := Native.statistics_eq Native.algebra (Native.equal I P) s X D c.block row c.tree _ (hx row hr)
    refine ⟨?_,?_,?_⟩
    · rw [effect.mean row (List.mem_range.mpr hr),stats]
      rfl
    · rw [effect.rstd row (List.mem_range.mpr hr),stats]
      rfl
    · intro col hc
      rw [effect.output row (List.mem_range.mpr hr) col hc]
      unfold rowOutput
      rw [stats]
      simp only [normalizeValue,hx row hr col hc,hw col hc,hb col hc]
      rfl
  · intro q off miss
    exact effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.KernelValues
