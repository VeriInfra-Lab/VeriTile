import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.RealBindings

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def programWrites (c : Host.Call) (idx : GridIndex c.grid) : WriteFootprint :=
  fun cell=>cell.1=c.regions.out ∧ ∃ i,c.active idx i ∧ c.offset idx i=cell.2

theorem one (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s)
    (idx : GridIndex c.grid) :
    ∃ f : Kernel.ExecFrame (source c).kernel.toAlgKernel (s.withGridIndex idx),
      f.writes=programWrites c idx ∧
      ∀ i,c.active idx i → f.final.readMem c.regions.out (c.offset idx i)=
        Formula.expected c.layout (c.launchAt idx) (Program.inputs s c.regions)
          (Host.scale c.layout.D) i.1.val i.2.1.val := by
  have active := Host.active_eq c A s valid idx
  have offset (i : RealIO.Lane c.layout) :
      EpilogueResult.offset c.layout c.regions (s.withGridIndex idx) i=c.offset idx i := by
    exact Host.offset_eq c A s valid idx i
  have injective : ∀ i j,EpilogueResult.activeLane c.layout c.regions (s.withGridIndex idx) i →
      EpilogueResult.activeLane c.layout c.regions (s.withGridIndex idx) j →
      EpilogueResult.offset c.layout c.regions (s.withGridIndex idx) i=
      EpilogueResult.offset c.layout c.regions (s.withGridIndex idx) j → i=j := by
    intro i j hi hj same
    rw [offset i,offset j] at same
    exact (Host.writer_unique c A s valid idx idx i j ((active i).mp hi) ((active j).mp hj) same).2
  obtain ⟨t,ran,result,frame⟩ := EpilogueResult.correct c.layout c.regions .fp32 Bool.false
    (Host.scale c.layout.D) (s.withGridIndex idx)
    (by rw [c.blockN]; decide)
    (by change 0 ≤ (Program.inputs s c.regions).start (c.launchAt idx).batch
        rw [valid.start_value _ (Host.batch_bound c idx)]; positivity) injective
  refine ⟨{
    final:=t
    writes:=programWrites c idx
    h_exec:=ran
    h_writeWithin:=?_ },rfl,?_⟩
  · intro r off outside
    symm
    apply frame r off
    rintro ⟨i,on,region,address⟩
    apply outside
    refine ⟨region.symm,i,(active i).mp on,?_⟩
    rwa [offset i] at address
  · intro i on
    have h:=result i ((active i).mpr on)
    rw [offset i] at h
    exact h

theorem whole (c : Host.Call) (A : FlatAlloc) (s : BlockState) (valid : Host.Valid c A s) :
    ∃ t,Nonempty (Kernel.GridLaunchedOrdinary (source c).kernel.toAlgKernel c.grid s t) ∧
      (∀ idx i,c.active idx i → t.readMem c.regions.out (c.offset idx i)=
        Formula.expected c.layout (c.launchAt idx) (Program.inputs s c.regions)
          (Host.scale c.layout.D) i.1.val i.2.1.val) ∧
      (∀ r off,(¬∃ idx,programWrites c idx (r,off)) → t.mem r off=s.mem r off) := by
  let frames : Kernel.GridFrames (source c).kernel.toAlgKernel c.grid s :=
    fun idx=>Classical.choose (one c A s valid idx)
  have writes (idx : GridIndex c.grid) : (frames idx).writes=programWrites c idx :=
    (Classical.choose_spec (one c A s valid idx)).1
  have separate : Kernel.GridWritesDisjoint frames := by
    intro left right different cell hl hr
    rw [writes left] at hl
    rw [writes right] at hr
    obtain ⟨_,i,hi,ei⟩:=hl
    obtain ⟨_,j,hj,ej⟩:=hr
    exact different (Host.writer_unique c A s valid left right i j hi hj (ei.trans ej.symm)).1
  refine ⟨Kernel.mergeFrames c.grid s frames,⟨frames,separate,rfl⟩,?_,?_⟩
  · intro idx i on
    have written : (frames idx).writes (c.regions.out,c.offset idx i) := by
      rw [writes idx]
      exact ⟨rfl,i,on,rfl⟩
    have memory:=Kernel.mergeFrames_mem_written separate idx c.regions.out (c.offset idx i) written
    simp only [BlockState.readMem,memory]
    exact (Classical.choose_spec (one c A s valid idx)).2 i on
  · intro r off outside
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨idx,hit⟩
    exact outside ⟨idx,by rwa [writes idx] at hit⟩

#axiomsClean one
#axiomsClean whole
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
