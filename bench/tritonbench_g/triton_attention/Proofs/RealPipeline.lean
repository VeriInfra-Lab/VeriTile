import bench.tritonbench_g.triton_attention.Proofs.RealBackwardGrid
import bench.tritonbench_g.triton_attention.Proofs.RealForwardGridSaved
import bench.tritonbench_g.triton_attention.Proofs.RealPreprocessGrid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def readReal (mem : Grid.Memory R) (r : RegionName) (o : Nat) : ℝ :=
  ((mem r o).read .fp32).getD 0

theorem readReal_of_read (mem : Grid.Memory R) (r : RegionName) (o : Nat) (x : ℝ)
    (h : (mem r o).read .fp32=some x) : readReal mem r o=x := by
  simp only [readReal,h,Option.getD_some]

theorem readReal_of_cell (mem : Grid.Memory R) (r : RegionName) (o : Nat) (x : ℝ)
    (h : mem r o=.mk .fp32 (some x)) : readReal mem r o=x := by
  rw [readReal,h]
  rfl

theorem row_coord (c : Shape) (o : Nat) (ho : o<c.groups*c.rows) :
    ∃ h : Fin c.groups,∃ i : Fin c.rows,rowOffset c h i=o := by
  refine ⟨⟨o/c.rows,(Nat.div_lt_iff_lt_mul c.rows_pos).mpr ho⟩,
    ⟨o%c.rows,Nat.mod_lt _ c.rows_pos⟩,?_⟩
  exact Nat.div_add_mod' _ _

theorem element_coord (c : Shape) (o : Nat) (ho : o<c.elements) :
    ∃ i : Coord c,offset c i=o := by
  have hr : o/c.dim<c.groups*c.rows := (Nat.div_lt_iff_lt_mul c.dim_pos).mpr ho
  obtain ⟨h,i,he⟩:=row_coord c (o/c.dim) hr
  refine ⟨(h,i,⟨o%c.dim,Nat.mod_lt _ c.dim_pos⟩),?_⟩
  change rowOffset c h i*c.dim+o%c.dim=o
  rw [he,Nat.div_add_mod']

theorem forwardStep_readonly (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (r : RegionName) (o : Nat) (hr : r≠"Out" ∧ r≠"L" ∧ r≠"M") :
    Grid.forwardStep model c scale pid mem r o=mem r o := by
  have hr' : r≠(forwardParameters c scale).output ∧ r≠(forwardParameters c scale).denominator ∧ r≠(forwardParameters c scale).maximum := hr
  unfold Grid.forwardStep forwardWrite
  dsimp only []
  rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inl hr'.1)]
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2.2)]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2.1) _

/-- Relations established by forward stores followed by preprocess stores.
All values in this record are derived during execution. -/
structure BackwardData (c : Shape) (scale : ℝ) (input saved : RegionName → Nat → ℝ) : Prop where
  inputs : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,saved r o=input r o
  denominator : ∀ h : Fin c.groups,∀ i : Fin c.rows,
    saved "L" (rowOffset c h i)=Mathematical.mass i.val c.rows
      (Mathematical.score c.dim scale (tensorAt c input "Q" h.val) (tensorAt c input "K" h.val) i.val)
      (saved "M" (rowOffset c h i))
  gradient : ∀ i : Coord c,saved "NewDO" (offset c i)=input "DO" (offset c i)/saved "L" (rowOffset c i.1 i.2.1)
  delta : ∀ h : Fin c.groups,∀ i : Fin c.rows,
    saved "Delta" (rowOffset c h i)=
      (∑ d ∈ Finset.range c.dim,Mathematical.attention c.dim scale
        (tensorAt c input "Q" h.val) (tensorAt c input "K" h.val) (tensorAt c input "V" h.val) i.val d *
          tensorAt c input "DO" h.val i.val d)/saved "L" (rowOffset c h i)

/-- The complete shared DSL invocation derives its own saved state and all
three backward outputs. Initial scratch values are arbitrary; only the DQ
zero initialization and the input loads are assumed. -/
theorem whole_execution_tiled (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory R) (μ : RegionName → Nat → ℝ)
    (inputs : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,(mem r o).read .fp32=some (μ r o))
    (zero : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (some 0)) :
    ∃ final saved,Grid.wholeRun algebra model model model c scale fw pw bw mem=some final ∧
      BackwardData c scale μ saved ∧
      (∀ i : Coord c,final "Out" (offset c i)=.mk .fp32 (some (outputReference c scale μ i)) ∧
        final "DQ" (offset c i)=.mk .fp32 (some (tiledQ c scale saved i)) ∧
        final "DK" (offset c i)=.mk .fp32 (some (tiledK c scale saved i)) ∧
        final "DV" (offset c i)=.mk .fp32 (some (tiledV c scale saved i))) ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  have hi : ForwardInputs mem μ := by
    intro r hr o
    exact inputs r (by simp only [List.mem_cons,List.not_mem_nil,or_false] at hr ⊢;tauto) o
  let f:=fw.foldl (fun m pid=>Grid.forwardStep model c scale pid m) mem
  have fRun : Grid.run (Grid.execute algebra model (forwardKernel c scale) (Grid.forwardGrid c)) fw mem=some f :=
    Grid.run_eq_fold _ _ (Grid.forward_success algebra model c scale) fw mem
  obtain ⟨f',fe,fi,fo⟩:=forward_grid_output c scale fw hf mem μ hi
  have ff : f'=f := Option.some.inj (fe.symm.trans fRun)
  subst f'
  have fs : ∀ i : Coord c,SavedAt c scale μ i f := forward_grid_saved c scale fw hf mem μ hi
  have fRead (r : RegionName) (o : Nat) (hr : r≠"Out" ∧ r≠"L" ∧ r≠"M") : f r o=mem r o :=
    Grid.fold_frame _ fw mem r o (by intro pid hpid m;exact forwardStep_readonly c scale pid m r o hr)
  let out:=fun row d=>readReal f "Out" (row*c.dim+d)
  let grad:=fun row d=>μ "DO" (row*c.dim+d)
  let den:=readReal f "L"
  have preData : PreData c f out grad den := by
    constructor
    · intro row hr d hd
      obtain ⟨h,i,he⟩:=row_coord c row hr
      have hh:=fo (h,i,⟨d,hd⟩)
      have hoff : offset c (h,i,⟨d,hd⟩)=row*c.dim+d := by change rowOffset c h i*c.dim+d=_;rw [he]
      rw [hoff] at hh
      rw [hh]
      change some _=some (readReal f "Out" _)
      rw [readReal_of_cell _ _ _ _ hh]
    constructor
    · intro row hr d hd
      rw [fRead "DO" _ (by decide)]
      exact inputs "DO" (by simp) _
    · intro row hr
      obtain ⟨h,i,he⟩:=row_coord c row hr
      obtain ⟨m,hm,hl⟩:=fs (h,i,⟨0,c.dim_pos⟩)
      rw [he] at hl
      rw [hl]
      change some _=some (readReal f "L" _)
      rw [readReal_of_cell _ _ _ _ hl]
  let p:=pw.foldl (fun m pid=>Grid.preprocessStep model c pid m) f
  have pRun : Grid.run (Grid.execute algebra model (preprocessKernel c) (Grid.preprocessGrid c)) pw f=some p :=
    Grid.run_eq_fold _ _ (Grid.preprocess_success algebra model c) pw f
  obtain ⟨p',pe,pi,po⟩:=preprocess_grid_values c pw hp f out grad den preData
  have pp : p'=p := Option.some.inj (pe.symm.trans pRun)
  subst p'
  have pRead (r : RegionName) (o : Nat) (hr : r≠"NewDO" ∧ r≠"Delta") : p r o=f r o :=
    Grid.fold_frame _ pw f r o (by intro pid hpid m;exact preprocessStep_readonly c pid m r o hr)
  let saved:=readReal p
  have sInputs : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,saved r o=μ r o := by
    intro r hr o
    apply readReal_of_read
    have ne : r≠"Out" ∧ r≠"L" ∧ r≠"M" ∧ r≠"NewDO" ∧ r≠"Delta" := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl <;> decide
    rw [pRead r o ne.2.2.2,fRead r o ⟨ne.1,ne.2.1,ne.2.2.1⟩]
    exact inputs r hr o
  have sL (row : Nat) : saved "L" row=den row := by
    change ((p "L" row).read .fp32).getD 0=((f "L" row).read .fp32).getD 0
    rw [pRead "L" row (by decide)]
  have out_eq (h : Fin c.groups) (i : Fin c.rows) (d : Fin c.dim) :
      out (rowOffset c h i) d.val=outputReference c scale μ (h,i,d) :=
    readReal_of_cell f "Out" _ _ (fo (h,i,d))
  have sData : BackwardData c scale μ saved := by
    refine ⟨sInputs,?_,?_,?_⟩
    · intro h i
      obtain ⟨m,hm,hl⟩:=fs (h,i,⟨0,c.dim_pos⟩)
      have sm : saved "M" (rowOffset c h i)=m := by
        apply readReal_of_cell
        rw [pRead "M" _ (by decide)]
        exact hm
      rw [sm,sL]
      exact readReal_of_cell f "L" _ _ hl
    · intro i
      rw [sL]
      exact readReal_of_cell p "NewDO" _ _ (po i).1
    · intro h i
      rw [sL]
      change readReal p "Delta" (rowOffset c h i)=_
      rw [readReal_of_cell p "Delta" _ _ (po (h,i,⟨0,c.dim_pos⟩)).2]
      congr 1
      apply Finset.sum_congr rfl
      intro d hd
      rw [out_eq h i ⟨d,Finset.mem_range.mp hd⟩]
      rfl
  have bReads : BackwardReads (backwardParameters c scale) p saved := by
    constructor
    · intro r hr o ho
      simp only [backwardParameters,List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl
      all_goals try {rw [sInputs _ (by simp) o];rw [pRead _ o (by decide),fRead _ o (by decide)];exact inputs _ (by simp) o}
      obtain ⟨i,he⟩:=element_coord c o ho
      rw [←he,(po i).1]
      change some _=some (saved "NewDO" _)
      exact congrArg some (readReal_of_cell p "NewDO" _ _ (po i).1).symm
    · intro r hr o ho
      simp only [backwardParameters,List.mem_cons,List.not_mem_nil,or_false] at hr
      obtain ⟨h,i,he⟩:=row_coord c o ho
      rw [←he]
      rcases hr with rfl|rfl
      · obtain ⟨m,hm,_⟩:=fs (h,i,⟨0,c.dim_pos⟩)
        have cell : p "M" (rowOffset c h i)=.mk .fp32 (some m) := by rw [pRead "M" _ (by decide)];exact hm
        rw [cell]
        exact congrArg some (readReal_of_cell p "M" _ _ cell).symm
      · rw [(po (h,i,⟨0,c.dim_pos⟩)).2]
        exact congrArg some (readReal_of_cell p "Delta" _ _ (po (h,i,⟨0,c.dim_pos⟩)).2).symm
  have pZero : ∀ i : Coord c,p "DQ" (offset c i)=.mk .fp32 (some 0) := by
    intro i
    rw [pRead "DQ" _ (by decide),fRead "DQ" _ (by decide)]
    exact zero i
  obtain ⟨b,be,bi,bo⟩:=backward_grid_values c scale bw hb p saved bReads pZero
  have whole : Grid.wholeRun algebra model model model c scale fw pw bw mem=some b := by
    unfold Grid.wholeRun
    rw [fRun]
    change (Grid.run (Grid.execute algebra model (preprocessKernel c) (Grid.preprocessGrid c)) pw f).bind
      (Grid.run (Grid.execute algebra model (backwardKernel c scale) (Grid.backwardGrid c)) bw)=some b
    rw [pRun]
    exact be
  refine ⟨b,saved,whole,sData,?_,?_⟩
  · intro i
    refine ⟨?_,bo i⟩
    have bb:=Grid.run_eq_fold _ _ (Grid.backward_success algebra model c scale) bw p
    have eqb : b=bw.foldl (fun m pid=>Grid.backwardStep model c scale pid m) p := Option.some.inj (be.symm.trans bb)
    rw [eqb,Grid.fold_frame _ bw p "Out" _ (by intro pid hpid m;exact Grid.backwardStep_readonly model c scale pid m "Out" _ (by decide))]
    rw [pRead "Out" _ (by decide)]
    exact fo i
  · have finalEq : b=Grid.wholeReference model model model c scale fw pw bw mem :=
      Option.some.inj (whole.symm.trans (Grid.whole_success algebra model model model c scale fw pw bw mem))
    rw [finalEq]
    exact Grid.wholeReference_frame model model model c scale fw pw bw mem
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)

#axiomsClean whole_execution_tiled
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
