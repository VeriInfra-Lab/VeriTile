import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelRealRaw
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.DirectLayout

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Actual ordinary main-grid execution in the mathematical Real projection of
one admitted FP32 invocation. There is no wrapper or allocator execution here. -/
def MainRuns (p : Metadata) (c : Host.Config) (initial final : BlockState) : Prop :=
  Nonempty (Kernel.GridLaunchedOrdinary
    (Kernels.fp32Source (call p) c).eraseDType.toAlgKernel ((call p).grid c) initial final)

/-- The complete kernel invocation, including every program of the actual
quoted cleanup DSL when causal. Both launches share the same Real memory. -/
def Runs (p : Metadata) (c : Host.Config) (initial final : BlockState) : Prop :=
  if (shape p).outputSize=0 then final=initial else
    ∃ middle, MainRuns p c initial middle ∧
      (if p.causal then CleanupReal.schedule (call p) Finset.univ.toList middle=some final
       else final=middle)

def OutputCell (p : Metadata) (region : RegionName) (offset : Nat) : Prop :=
  region=(call p).Out ∧ offset<(shape p).outputSize

/-- Construct the actual main grid from the existing per-program source
executions. Its values and frame are derived from those executions. -/
theorem main_complete (p : Metadata) (c : Host.Config) (hc : c∈Host.configurations)
    (x : Host.Inputs (call p)) (s : BlockState) (pins : Host.Pinned (call p) c x s) :
    ∃ t, MainRuns p c s t ∧
      (∀ idx i,Host.active (call p) c idx i →
        t.mem (call p).Out (Host.offset (call p) c idx i)=
          MemCell.real (Host.value (call p) c x idx i)) ∧
      (∀ r off,¬OutputCell p r off → t.mem r off=s.mem r off)  :=
  Raw.main_complete (call p) c hc rfl rfl x s pins

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

theorem product_link (p : Metadata) (x : Mathematical.Inputs) (b ch g r n : Nat) :
    Mathematical.product p x b ch g r n=
      if p.causal=Bool.true ∧ r<n then 0 else Host.expected (call p) (Direct.realInputs p x) b ch g r n := by
  simp only [Mathematical.product,Host.expected,call,Direct.realInputs,
    Mathematical.inputIndex,Direct.inputIndex,Nat.add_assoc]
  rfl

theorem offset_bound (p : Metadata) (b ch g r n : Nat)
    (hb : b<(shape p).batch) (hch : ch<(shape p).chunks) (hg : g<(shape p).groups)
    (hr : r<(shape p).chunk) (hn : n<(shape p).chunk) :
    Host.returnedOffset (call p) b ch g r n<(shape p).outputSize := by
  have bound := Host.mul_add_bound (Host.mul_add_bound
    (Host.mul_add_bound (Host.mul_add_bound hb hch) hg) hr) hn
  convert bound using 1 <;> simp only [Host.returnedOffset,Host.Shape.outputSize,call] <;> ring

/-- Kernel-only arbitrary-Real result: source main-grid and actual cleanup-grid
success, the independent full product, and every cell outside the output. -/
theorem complete (p : Metadata) (h : InputContract p) (x : Mathematical.Inputs)
    (c : Host.Config) (hc : c∈Host.configurations) :
    ∃ final,Runs p c (Direct.realState p x) final ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        final.readMem (call p).Out (Host.returnedOffset (call p) b ch g r n)=
          Mathematical.product p x b ch g r n) ∧
      (∀ r off,¬OutputCell p r off → final.mem r off=(Direct.realState p x).mem r off) := by
  by_cases empty : (shape p).outputSize=0
  · refine ⟨Direct.realState p x, by simp [Runs,empty], ?_, fun _ _ _ => rfl⟩
    intro b ch g r n hb hch hg hr hn
    have bound := offset_bound p b ch g r n hb hch hg hr hn
    omega
  · have binding := Direct.real_binding p h x (by omega)
    obtain ⟨final,runs,values,frame⟩ := Raw.complete (call p) c hc rfl rfl
      (Direct.realInputs p x) (Direct.realState p x) (Host.tensor_pins _ _ _ _ _ binding)
    refine ⟨final,runs,?_,frame⟩
    intro b ch g r n hb hch hg hr hn
    exact (values b ch g r n hb hch hg hr hn).trans (product_link p x b ch g r n).symm

#axiomsClean main_complete
#axiomsClean main_value
#axiomsClean complete
#specNonCircular Mathematical.product avoiding [Runs,MainRuns,CleanupReal.schedule,Kernels.source]
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelReal
