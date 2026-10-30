import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.RealAddressBounds
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOOwnership

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def tileActive (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values r extent)
    (idx : GridIndex (Host.grid g.arguments)) (i : RealIO.Lane (Host.layout g.arguments)) : Prop :=
  RealIO.active (Host.layout g.arguments) (Host.launchOf g.arguments idx) (mathInputs r extent xs) i

def tileOffset (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values r extent)
    (idx : GridIndex (Host.grid g.arguments)) (i : RealIO.Lane (Host.layout g.arguments)) : Nat :=
  RealIO.write (Host.layout g.arguments) (Host.launchOf g.arguments idx) (mathInputs r extent xs) i

def tileResult (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values r extent)
    (idx : GridIndex (Host.grid g.arguments)) (i : RealIO.Lane (Host.layout g.arguments)) : ℝ :=
  Formula.expected (Host.layout g.arguments) (Host.launchOf g.arguments idx) (mathInputs r extent xs)
    (Host.scale g.arguments) i.1.val i.2.1.val
    (Math.count (Host.layout g.arguments) (Host.launchOf g.arguments idx) (mathInputs r extent xs))

theorem launch_initial (g : Raw.Geometry) (idx : GridIndex (Host.grid g.arguments)) (s : BlockState) :
    (launch g idx).initial s=s.withGridIndex idx := rfl

theorem active_matches (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (idx : GridIndex (Host.grid g.arguments))
    (i : RealIO.Lane (Host.layout g.arguments)) :
    EpilogueResult.activeLane (Host.layout g.arguments) r (s.withGridIndex idx) i ↔ tileActive g r extent xs idx i := by
  have agree := input_agreement g r extent xs valid s hin idx
  simp only [EpilogueResult.activeLane,Host.launch_grid,Host.inputs_grid,agree.sequence_eq,tileActive,RealIO.active]

theorem offset_matches (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (idx : GridIndex (Host.grid g.arguments))
    (i : RealIO.Lane (Host.layout g.arguments)) :
    EpilogueResult.offset (Host.layout g.arguments) r (s.withGridIndex idx) i=tileOffset g r extent xs idx i := by
  have agree := input_agreement g r extent xs valid s hin idx
  change RealIO.write (Host.layout g.arguments) (Host.launchOf g.arguments idx) (inputs s r) i=_
  simp only [tileOffset,RealIO.write,AddressBounds.outputAddress,agree.start]

theorem stage_safe (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (bounds : RegionBounds)
    (hb : (io g r extent).InBounds bounds s xs) (idx : GridIndex (Host.grid g.arguments)) :
    (AlgorithmStage.single (source g r) (launch g idx)).Safe bounds s := by
  have agree := input_agreement g r extent xs valid s hin idx
  have prompt : 0≤(inputs (s.withGridIndex idx) r).prompt ((s.withGridIndex idx).pids 0) := by
    change 0≤(inputs s r).prompt (Host.launchOf g.arguments idx).batch
    rw [agree.prompt]
    exact (valid.metadata _ (Host.batch_lt g.arguments idx)).2.1
  exact WholeSafety.complete (Host.layout g.arguments) r .fp32 Bool.false (Host.scale g.arguments) (s.withGridIndex idx) (Host.block_positive g.arguments) prompt bounds
    (by simpa only [Host.launch_grid,Host.inputs_grid] using actual_bounds g r extent xs valid s hin bounds hb idx)

theorem stage_correct (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (idx : GridIndex (Host.grid g.arguments)) :
    ∃ t,(AlgorithmStage.single (source g r) (launch g idx)).Runs s t ∧
      (∀ i,tileActive g r extent xs idx i →
        t.readMem r.out (tileOffset g r extent xs idx i)=tileResult g r extent xs idx i) ∧
      (∀ rg off,(∀ i,tileActive g r extent xs idx i → rg≠r.out ∨ off≠tileOffset g r extent xs idx i) →
        t.mem rg off=s.mem rg off) := by
  have agree := input_agreement g r extent xs valid s hin idx
  have prompt : 0≤(inputs (s.withGridIndex idx) r).prompt ((s.withGridIndex idx).pids 0) := by
    change 0≤(inputs s r).prompt (Host.launchOf g.arguments idx).batch
    rw [agree.prompt]
    exact (valid.metadata _ (Host.batch_lt g.arguments idx)).2.1
  have unique : ∀ i j,EpilogueResult.activeLane (Host.layout g.arguments) r (s.withGridIndex idx) i →
      EpilogueResult.activeLane (Host.layout g.arguments) r (s.withGridIndex idx) j →
      EpilogueResult.offset (Host.layout g.arguments) r (s.withGridIndex idx) i=
        EpilogueResult.offset (Host.layout g.arguments) r (s.withGridIndex idx) j → i=j := by
    intro i j hi hj same
    have hi' := (active_matches g r extent xs valid s hin idx i).mp hi
    have hj' := (active_matches g r extent xs valid s hin idx j).mp hj
    rw [offset_matches g r extent xs valid s hin idx,offset_matches g r extent xs valid s hin idx] at same
    exact (KernelIO.unique_writer g.arguments r extent (mathInputs r extent xs) valid idx idx i j hi' hj' same).2
  obtain ⟨t,ran,values,frame⟩ := EpilogueResult.correct (Host.layout g.arguments) r .fp32 Bool.false (Host.scale g.arguments)
    (s.withGridIndex idx) (Host.block_positive g.arguments) prompt unique
  refine ⟨t,(AlgorithmStage.single_runs _ _ _ _).mpr ran,?_,?_⟩
  · intro i active
    have result := values i ((active_matches g r extent xs valid s hin idx i).mpr active)
    rw [offset_matches g r extent xs valid s hin idx,WholeExecution.output,Host.launch_grid,Host.inputs_grid,
      agree.count_eq,RealInputAgreement.expected_eq _ r extent _ _ _ agree (valid.bounds idx)
        (Host.block_positive g.arguments) _ i.1 i.2.1] at result
    exact result
  · intro rg off outside
    apply frame
    rintro ⟨i,active,hr,ho⟩
    have hi := (active_matches g r extent xs valid s hin idx i).mp active
    rw [offset_matches g r extent xs valid s hin idx] at ho
    exact (outside i hi).elim (fun different=>different hr.symm) (fun different=>different ho.symm)

theorem input_output_separate (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (i : Fin 8) : inputRegion r i≠r.out := by
  have h:=valid.separate
  fin_cases i <;> simp only [inputRegion] <;> tauto

theorem binding_preserved (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s t : BlockState)
    (hin : (io g r extent).InputBinding s xs)
    (frame : ∀ rg off,rg≠r.out → t.mem rg off=s.mem rg off) :
    (io g r extent).InputBinding t xs := by
  intro i j active
  have same := frame (inputRegion r i) j.val (input_output_separate g r extent xs valid i)
  have bound := hin i j active
  change (inputType i).read t (inputRegion r i) j.val=xs i j
  change (inputType i).read s (inputRegion r i) j.val=xs i j at bound
  have reads : ∀ ty : ChanTy,ty.read t (inputRegion r i) j.val=ty.read s (inputRegion r i) j.val := by
    intro ty
    cases ty <;> simp only [ChanTy.read,BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped,same]
  rw [reads]
  exact bound

#axiomsClean stage_correct
#axiomsClean stage_safe
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
