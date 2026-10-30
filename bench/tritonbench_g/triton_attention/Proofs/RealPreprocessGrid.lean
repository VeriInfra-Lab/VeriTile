import bench.tritonbench_g.triton_attention.Proofs.RealPreprocess
import bench.tritonbench_g.triton_attention.Proofs.PreGridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def PreData (c : Shape) (mem : Grid.Memory R) (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) : Prop :=
  (∀ row<c.groups*c.rows,∀ d<c.dim,(mem "Out" (row*c.dim+d)).read .fp32=some (out row d)) ∧
  (∀ row<c.groups*c.rows,∀ d<c.dim,(mem "DO" (row*c.dim+d)).read .fp32=some (grad row d)) ∧
  (∀ row<c.groups*c.rows,(mem "L" row).read .fp32=some (den row))

theorem preData_reads (c : Shape) (pid : Grid.Program) (mem : Grid.Memory R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreData c mem out grad den)
    (hp : Grid.Legal (Grid.preprocessGrid c) pid) :
    PreReads (preParameters c) (Grid.entry (Grid.preprocessGrid c) pid mem) out grad den := by
  constructor
  · intro i
    exact h.1 _ (Grid.preRow_bound c pid mem hp (i.1,())) _ i.2.1.isLt
  constructor
  · intro i
    exact h.2.1 _ (Grid.preRow_bound c pid mem hp (i.1,())) _ i.2.1.isLt
  · intro i
    exact h.2.2 _ (Grid.preRow_bound c pid mem hp i)

theorem preprocessStep_readonly (c : Shape) (pid : Grid.Program) (mem : Grid.Memory R)
    (r : RegionName) (o : Nat) (hr : r≠"NewDO" ∧ r≠"Delta") :
    Grid.preprocessStep model c pid mem r o=mem r o := by
  have hr' : r≠(preParameters c).newGradient ∧ r≠(preParameters c).delta := hr
  unfold Grid.preprocessStep preWrite
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.2)]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inl hr'.1) _

theorem preprocessStep_data (c : Shape) (pid : Grid.Program) (mem : Grid.Memory R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreData c mem out grad den) :
    PreData c (Grid.preprocessStep model c pid mem) out grad den := by
  constructor
  · intro row hr d hd
    rw [preprocessStep_readonly c pid mem "Out" _ (by decide)]
    exact h.1 row hr d hd
  constructor
  · intro row hr d hd
    rw [preprocessStep_readonly c pid mem "DO" _ (by decide)]
    exact h.2.1 row hr d hd
  · intro row hr
    rw [preprocessStep_readonly c pid mem "L" row (by decide)]
    exact h.2.2 row hr

theorem preprocessStep_values (c : Shape) (mem : Grid.Memory R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreData c mem out grad den) (i : Coord c) :
    Grid.preprocessStep model c (Grid.preProgram c i) mem "NewDO" (offset c i)=.mk .fp32
      (some (grad (rowOffset c i.1 i.2.1) i.2.2.val/den (rowOffset c i.1 i.2.1))) ∧
    Grid.preprocessStep model c (Grid.preProgram c i) mem "Delta" (rowOffset c i.1 i.2.1)=.mk .fp32
      (some ((∑ d ∈ Finset.range c.dim,out (rowOffset c i.1 i.2.1) d*grad (rowOffset c i.1 i.2.1) d)/den (rowOffset c i.1 i.2.1))) := by
  let pid:=Grid.preProgram c i
  let s:=Grid.entry (Grid.preprocessGrid c) pid mem
  have hp:=Grid.preProgram_legal c i
  have reads:=preData_reads c pid mem out grad den h hp
  obtain ⟨final,_,hm,hg,hd,_⟩:=preprocessCorrectness algebra model (preParameters c) s
    (by simp [preParameters]) (Grid.preOffset_injective c pid mem)
  have a:=hg (Grid.preLane c i)
  have b:=hd ((Grid.preLane c i).1,())
  rw [hm] at a b
  change Grid.preprocessStep model c pid mem "NewDO" (preOffset (preParameters c) s (Grid.preLane c i))=_ at a
  change Grid.preprocessStep model c pid mem "Delta" (preRow (preParameters c) s ((Grid.preLane c i).1,()))=_ at b
  dsimp only [s,pid] at a b
  rw [Grid.pre_target_offset] at a
  rw [Grid.pre_target_row] at b
  change Grid.preprocessStep model c (Grid.preProgram c i) mem "NewDO" (offset c i)=
    Cell.mk .fp32 (preQuotient model (preParameters c) s (Grid.preLane c i)) at a
  change Grid.preprocessStep model c (Grid.preProgram c i) mem "Delta" (rowOffset c i.1 i.2.1)=
    Cell.mk .fp32 (preDelta model (preParameters c) s ((Grid.preLane c i).1,())) at b
  rw [preQuotient_real (preParameters c) s out grad den reads] at a
  rw [preDelta_real (preParameters c) s out grad den reads] at b
  dsimp only [s,pid] at a b
  rw [Grid.pre_target_row] at a b
  exact ⟨a,b⟩

theorem preprocess_grid_values (c : Shape) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.preprocessGrid c) work) (mem : Grid.Memory R)
    (out grad : Nat → Nat → ℝ) (den : Nat → ℝ) (h : PreData c mem out grad den) :
    ∃ final,Grid.run (Grid.execute algebra model (preprocessKernel c) (Grid.preprocessGrid c)) work mem=some final ∧
      PreData c final out grad den ∧
      ∀ i : Coord c,
        final "NewDO" (offset c i)=.mk .fp32 (some (grad (rowOffset c i.1 i.2.1) i.2.2.val/den (rowOffset c i.1 i.2.1))) ∧
        final "Delta" (rowOffset c i.1 i.2.1)=.mk .fp32 (some
          ((∑ d ∈ Finset.range c.dim,out (rowOffset c i.1 i.2.1) d*grad (rowOffset c i.1 i.2.1) d)/den (rowOffset c i.1 i.2.1))) := by
  let final:=work.foldl (fun m pid=>Grid.preprocessStep model c pid m) mem
  have hinv : PreData c final out grad den :=
    Grid.fold_invariant (Grid.preprocessStep model c) (fun m=>PreData c m out grad den) work mem h
      (by intro pid hp m hm;exact preprocessStep_data c pid m out grad den hm)
  refine ⟨final,Grid.run_eq_fold _ _ (Grid.preprocess_success algebra model c) work mem,hinv,?_⟩
  intro i
  constructor
  · exact Grid.fold_readback (Grid.preprocessStep model c) (fun m=>PreData c m out grad den) work mem
      (Grid.preProgram c i) "NewDO" (offset c i) _
      (Grid.FullSchedule.covers _ work schedule _ (Grid.preProgram_legal c i)) h
      (by intro pid hp m hm;exact preprocessStep_data c pid m out grad den hm)
      (by intro m hm;exact (preprocessStep_values c m out grad den hm i).1)
      (by intro pid hp hn m;exact Grid.preprocessStep_otherNewDO model c pid m (Grid.FullSchedule.legal _ work schedule pid hp) i hn)
  · exact Grid.fold_readback (Grid.preprocessStep model c) (fun m=>PreData c m out grad den) work mem
      (Grid.preProgram c i) "Delta" (rowOffset c i.1 i.2.1) _
      (Grid.FullSchedule.covers _ work schedule _ (Grid.preProgram_legal c i)) h
      (by intro pid hp m hm;exact preprocessStep_data c pid m out grad den hm)
      (by intro m hm;exact (preprocessStep_values c m out grad den hm i).2)
      (by intro pid hp hn m;exact Grid.preprocessStep_otherDelta model c pid m (Grid.FullSchedule.legal _ work schedule pid hp) i hn)

#axiomsClean preprocess_grid_values
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
