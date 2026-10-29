import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.ReferenceBinding
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IOSafety
open VeriTile Triton FP.Structural TokenAttnMistral Binding
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem bounds_congr (p : Parameters) (s t : State α) (bounds : RegionName → Nat)
    (inside : AccessSafety.Bounds p s bounds) (memory : t.mem=s.mem) (pids : t.pids=s.pids) :
    AccessSafety.Bounds p t bounds := by
  constructor
  · simpa only [pids] using inside.sequence
  · simpa only [pids] using inside.start
  · simpa only [pids] using inside.request
  · simpa only [pids] using inside.count
  · simpa only [probOffset,count,startIndex,memory,pids] using inside.probability
  · simpa only [gatherOffset,count,windowStart,sequenceLength,requestIndex,memory,pids] using inside.tokens
  · simpa only [valueOffset,gather,gatherOffset,count,windowStart,sequenceLength,requestIndex,memory,pids] using inside.value
  · simpa only [outputOffset,pids] using inside.output

theorem grid_bounds (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) (vals : Ports.Values α p work counts)
    (bounds : RegionName → Nat) (hb : (IO.program M R p work counts).InBounds bounds vals)
    (s : State α) (f : Facts p work counts (Ports.logical p work counts vals) s.mem)
    (j : Fin work.length) : AccessSafety.Bounds p (Grid.state s.mem work[j]) bounds := by
  constructor
  · simpa [IO.program,Ports.memory,Ports.inputs,Binding.state_mem,Binding.state_pid0,Binding.state_pid1] using
      hb.2.1 ⟨0,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · simpa [IO.program,Ports.memory,Ports.inputs,Binding.state_mem,Binding.state_pid0,Binding.state_pid1] using
      hb.2.1 ⟨1,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · simpa [IO.program,Ports.memory,Ports.inputs,Binding.state_mem,Binding.state_pid0,Binding.state_pid1] using
      hb.2.1 ⟨2,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · simpa [IO.program,Ports.memory,Ports.inputs,Binding.state_mem,Binding.state_pid0,Binding.state_pid1] using
      hb.2.1 ⟨3,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · intro pos n live
    rw [f.count] at live
    have hspan : n<Ports.span p work counts := by have:=count_le p work counts j; unfold Ports.span; omega
    have h := hb.2.1 ⟨5,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,⟨n,hspan⟩,())
      (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro pos live)
    change work[j].2*p.probHeadStride+((Ports.logical p work counts vals).start j+n)*p.probTokenStride<bounds p.prob at h
    rw [probOffset,f.startIndex j]
    exact h
  · intro n active aligned lane live
    rw [f.count] at active
    simp only [f.windowStart,f.sequenceLength] at live
    have hspan := lane_bound p work counts j n active lane
    have visit : Ports.visited p (counts j) (n+lane.val) :=
      ⟨by have:=lane.isLt; omega,by rw [visited_lane p n aligned lane]; exact active⟩
    have h := hb.2.1 ⟨4,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,⟨n+lane.val,hspan⟩,())
      (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro visit live)
    change (Ports.logical p work counts vals).request j*p.requestBatchStride+
      (Ports.window p (Ports.logical p work counts vals) j+(n+lane.val))*p.requestTokenStride<bounds p.tokens.cast at h
    simpa only [gatherOffset,f.requestIndex,f.windowStart] using h
  · intro n active aligned lane live d
    rw [f.count] at active
    simp only [f.windowStart,f.sequenceLength] at live
    have hspan := lane_bound p work counts j n active lane
    have visit : Ports.visited p (counts j) (n+lane.val) :=
      ⟨by have:=lane.isLt; omega,by rw [visited_lane p n aligned lane]; exact active⟩
    have eq : gatherOffset p (Grid.state s.mem work[j]) lane.val+n*p.requestTokenStride=
        (Ports.logical p work counts vals).request j*p.requestBatchStride+
          (Ports.window p (Ports.logical p work counts vals) j+(n+lane.val))*p.requestTokenStride := by
      simp only [gatherOffset,f.requestIndex,f.windowStart]; ring
    have token := f.tokens j ⟨n+lane.val,hspan⟩ visit live
    have h := hb.2.1 ⟨6,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,⟨n+lane.val,hspan⟩,d,())
      (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro visit live)
    change work[j].2/p.kvGroup*p.valueHeadStride+d.val*p.valueDimStride+
      (Ports.logical p work counts vals).token j ⟨n+lane.val,hspan⟩*p.valueBatchStride<bounds p.value at h
    simpa only [valueOffset,gather,f.windowStart,f.sequenceLength,eq,live,if_true,
      Binding.state_mem,Binding.state_pid1,token] using h
  · intro d
    simpa [IO.program,Ports.memory,Ports.outputs,outputOffset,Grid.offset,Binding.state_mem,Binding.state_pid0,Binding.state_pid1] using
      hb.2.2.1 ⟨0,by simp [IO.program,Ports.memory,Ports.outputs]⟩ (j,d,()) (by change True; trivial)

theorem schedule (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) (vals : Ports.Values α p work counts)
    (bounds : RegionName → Nat) (hb : (IO.program M R p work counts).InBounds bounds vals)
    (separate : Grid.Separate p) (jobs : List Grid.PID) (member : ∀ pid∈jobs,pid∈work)
    (s : State α) (f : Facts p work counts (Ports.logical p work counts vals) s.mem) :
    ScheduleSafe bounds (jobs.map (IO.launch M R p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    obtain ⟨j,hj⟩ := List.mem_iff_get.mp (member pid (by simp))
    change work[j]=pid at hj
    subst pid
    constructor
    · apply AccessSafety.complete
      exact bounds_congr p _ _ bounds (grid_bounds M R p work counts vals bounds hb s f j) rfl rfl
    · intro t ran
      exact ih (fun pid hp=>member pid (by simp [hp])) t
        (f.congr (IOExecution.launch_agree M R p work[j] s t separate ran))

#axiomsClean schedule
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.IOSafety
