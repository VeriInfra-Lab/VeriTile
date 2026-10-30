import bench.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupReal
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.HostReturned

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelReal.Raw
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Actual ordinary main-grid execution in the mathematical Real projection of
one admitted FP32 invocation. There is no wrapper or allocator execution here. -/
def MainRuns (h : Host.Call) (c : Host.Config) (initial final : BlockState) : Prop :=
  Nonempty (Kernel.GridLaunchedOrdinary
    (Kernels.fp32Source h c).eraseDType.toAlgKernel (h.grid c) initial final)

/-- The complete kernel invocation, including every program of the actual
quoted cleanup DSL when causal. Both launches share the same Real memory. -/
def Runs (h : Host.Call) (c : Host.Config) (initial final : BlockState) : Prop :=
  if h.shape.outputSize=0 then final=initial else
    ∃ middle, MainRuns h c initial middle ∧
      (if h.causal then CleanupReal.schedule h Finset.univ.toList middle=some final
       else final=middle)

def OutputCell (h : Host.Call) (region : RegionName) (offset : Nat) : Prop :=
  region=h.Out ∧ offset<h.shape.outputSize

/-- Construct the actual main grid from the existing per-program source
executions. Its values and frame are derived from those executions. -/
theorem main_complete (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32)
    (x : Host.Inputs h) (s : BlockState) (pins : Host.Pinned h c x s) :
    ∃ t, MainRuns h c s t ∧
      (∀ idx i,Host.active h c idx i →
        t.mem h.Out (Host.offset h c idx i)=
          MemCell.real (Host.value h c x idx i)) ∧
      (∀ r off,¬OutputCell h r off → t.mem r off=s.mem r off) := by
  obtain ⟨frames,hw,hv⟩ := Host.frames_exist h c hc Bool.false x s pins
  have hd : Kernel.GridWritesDisjoint frames := by
    intro a b hab
    rw [hw a,hw b]
    exact Host.writes_disjoint h c hc a b hab
  let t := Kernel.mergeFrames (h.grid c) s frames
  have erase : (Kernels.fp32Source h c).eraseDType=
      RealProgram.kernel Bool.false c.M c.N c.K h.parameters := by
    rw [Kernels.fp32Source_eq h c ha hb]
    exact RealProgram.storage_erasure h.aStorage h.bStorage Bool.false
      (Host.outputDType h.aStorage h.requestedOutput) c.M c.N c.K h.parameters
  refine ⟨t,?_,?_,?_⟩
  · unfold MainRuns
    rw [erase]
    exact ⟨⟨frames,hd,rfl⟩⟩
  · intro idx i hi
    have mem := Kernel.mergeFrames_mem_written hd idx h.Out (Host.offset h c idx i)
      (by rw [hw]; exact ⟨rfl,i,hi,rfl⟩)
    exact mem.trans (hv idx i hi)
  · intro r off outside
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨idx,hi⟩
    rw [hw idx] at hi
    obtain ⟨rfl,i,hi,eq⟩ := hi
    apply outside
    refine ⟨rfl,?_⟩
    change Host.offset h c idx i=off at eq
    rw [←eq]
    exact Host.output_bound h c idx i hi

/-- Logical coordinates of every main-grid store, including partial chunks. -/
theorem main_value (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (x : Host.Inputs h) (t : BlockState)
    (values : ∀ idx i,Host.active h c idx i →
      t.mem h.Out (Host.offset h c idx i)=MemCell.real (Host.value h c x idx i))
    (b ch g r n : Nat) (hb : b<h.shape.batch) (hch : ch<h.shape.chunks)
    (hg : g<h.shape.groups) (hr : r<h.shape.chunk) (hn : n<h.shape.chunk)
    (active : Host.returnedActive h c r n) :
    t.readMem h.Out (Host.returnedOffset h b ch g r n)=Host.expected h x b ch g r n := by
  obtain ⟨idx,i,hpb,hpg,hpr,hpn⟩ := Host.coverage h c hc b ch g r n hb hch hg hr hn
  have hm := (Host.configuration_positive c hc).1
  have hn' := (Host.configuration_positive c hc).2.1
  have hgroups : 0<h.shape.groups := Nat.zero_lt_of_lt hg
  have hpc : Host.p2 idx/h.shape.groups=ch := by
    rw [hpg,Nat.mul_comm ch,Nat.mul_add_div hgroups]
    simp [Nat.div_eq_of_lt hg]
  have hph : Host.p2 idx%h.shape.groups=g := by
    rw [hpg]
    simp [Nat.add_mod,Nat.mod_eq_of_lt hg]
  have hpm : Host.p0 idx/RealSetup.tilesN c.N h.parameters=r/c.M := by
    rw [←hpr]
    unfold RealViews.row
    rw [Nat.mul_comm _ c.M,Nat.mul_add_div hm]
    simp [Nat.div_eq_of_lt i.1.isLt]
  have hpn' : Host.p0 idx%RealSetup.tilesN c.N h.parameters=n/c.N := by
    rw [←hpn]
    unfold RealViews.column
    rw [Nat.mul_comm _ c.N,Nat.mul_add_div hn']
    simp [Nat.div_eq_of_lt i.2.1.isLt]
  have hi : Host.active h c idx i := by
    refine ⟨?_,hpr ▸ hr,hpn ▸ hn⟩
    change (!h.causal || decide (Host.p0 idx%RealSetup.tilesN c.N h.parameters*c.N <
      (Host.p0 idx/RealSetup.tilesN c.N h.parameters+1)*c.M))=Bool.true
    rw [hpm,hpn']
    simpa [Host.returnedActive] using active
  have hv := values idx i hi
  have offset : Host.offset h c idx i=Host.returnedOffset h b ch g r n := by
    unfold Host.offset
    rw [Host.output_offset,hpb,hpg,hpr,hpn]
    unfold Host.returnedOffset
    ring
  rw [offset] at hv
  simp only [BlockState.readMem,hv,Host.value,hpb,hpc,hph,hpr,hpn]
  rfl

theorem offset_bound (h : Host.Call) (b ch g r n : Nat)
    (hb : b<h.shape.batch) (hch : ch<h.shape.chunks) (hg : g<h.shape.groups)
    (hr : r<h.shape.chunk) (hn : n<h.shape.chunk) :
    Host.returnedOffset h b ch g r n<h.shape.outputSize := by
  have bound := Host.mul_add_bound (Host.mul_add_bound
    (Host.mul_add_bound (Host.mul_add_bound hb hch) hg) hr) hn
  convert bound using 1 <;> simp only [Host.returnedOffset,Host.Shape.outputSize] <;> ring

/-- Kernel-only arbitrary-Real result: source main-grid and actual cleanup-grid
success, the independent full product, and every cell outside the output. -/
theorem complete (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) (x : Host.Inputs h)
    (s : BlockState) (pins : Host.Pinned h c x s) :
    ∃ final,Runs h c s final ∧
      (∀ b ch g r n,b<h.shape.batch → ch<h.shape.chunks → g<h.shape.groups →
        r<h.shape.chunk → n<h.shape.chunk →
        final.readMem h.Out (Host.returnedOffset h b ch g r n)=
          (if h.causal=Bool.true ∧ r<n then 0 else Host.expected h x b ch g r n)) ∧
      (∀ r off,¬OutputCell h r off → final.mem r off=s.mem r off) := by
  by_cases empty : h.shape.outputSize=0
  · refine ⟨s,by simp [Runs,empty],?_,fun _ _ _ => rfl⟩
    intro b ch g r n hb hch hg hr hn
    have bound := offset_bound h b ch g r n hb hch hg hr hn
    omega
  · obtain ⟨middle,run,values,frame⟩ := main_complete h c hc ha hb x s pins
    cases causal : h.causal
    · refine ⟨middle,?_,?_,frame⟩
      · simp only [Runs,if_neg empty,causal,Bool.false_eq_true,↓reduceIte]
        exact ⟨middle,run,rfl⟩
      · intro b ch g r n hb hch hg hr hn
        simp only [Bool.false_eq_true,false_and,↓reduceIte]
        exact main_value h c hc _ middle values b ch g r n hb hch hg hr hn (Or.inl causal)
    · obtain ⟨final,cleanup,mem⟩ := CleanupReal.complete h middle
      refine ⟨final,?_,?_,?_⟩
      · simp only [Runs,if_neg empty,causal,↓reduceIte]
        exact ⟨middle,run,cleanup⟩
      · intro b ch g r n hb hch hg hr hn
        have offset : Host.returnedOffset h b ch g r n=
            ((b*(h.shape.chunks*h.shape.groups)+(ch*h.shape.groups+g))*h.shape.chunk+r)*
              h.shape.chunk+n := by
          unfold Host.returnedOffset
          change _=(((b*(h.shape.chunks*h.shape.groups)+
            (ch*h.shape.groups+g))*h.shape.chunk+r)*h.shape.chunk+n)
          ring
        have upper : Cleanup.Upper h (Host.returnedOffset h b ch g r n) ↔ r<n := by
          rw [offset]
          exact Cleanup.upper_at h b (ch*h.shape.groups+g) r n hb
            (Host.mul_add_bound hch hg) hr hn
        unfold BlockState.readMem
        rw [mem]
        simp only [↓reduceIte,Cleanup.run_value]
        by_cases triangle : r<n
        · simp only [if_pos (upper.mpr triangle),triangle,and_self,↓reduceIte]
          rfl
        · simp only [if_neg (fun hu => triangle (upper.mp hu)),triangle,and_false,↓reduceIte]
          exact main_value h c hc _ middle values b ch g r n hb hch hg hr hn
            (Cleanup.lower_active h c hc r n (by omega))
      · intro r off outside
        rw [mem]
        by_cases same : r=h.Out
        · subst r
          rw [if_pos rfl,Cleanup.run_value,if_neg ?_]
          · exact frame _ _ outside
          · intro upper
            obtain ⟨idx,i,j,valid,eq⟩ := (Cleanup.member h off).mp
              ((Cleanup.image h off).mpr upper)
            apply outside
            refine ⟨rfl,?_⟩
            rw [eq]
            exact Cleanup.offset_bound h idx i j valid
        · rw [if_neg same]
          exact frame _ _ outside

#axiomsClean main_complete
#axiomsClean main_value
#axiomsClean complete
#specNonCircular Host.expected avoiding [Runs,MainRuns,CleanupReal.schedule,Kernels.source]
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelReal.Raw
