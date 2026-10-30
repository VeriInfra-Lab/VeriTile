import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealGrid
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.WholeReal
open VeriTile.Triton Layout
open scoped BigOperators
noncomputable section

def read (cell : MemCell) : ℝ := match cell.readAs .real with
  | some (some v) => v
  | _ => 0

def entry (c : Shape) (p : Storage.Placement c) (padding : RegionName → Nat → ℝ)
    (heap : Nat → MemCell) : BlockState where
  mem := Storage.lift c p (MemCell.real 0) heap
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax=0 then c.B*c.H else 1
  undef := padding

def source (c : Shape) (p : Storage.Placement c) : Host.Source c BlockState MemCell where
  input := fun s i => s.mem p.input (offset c i)
  output := fun s i => s.mem p.output (offset c i)
  launch := fun cfg => RealGrid.run p.input p.output c.T (c.B*c.H) cfg (List.range (c.B*c.H))

def Frame (c : Shape) (p : Storage.Placement c) := RealGrid.Frame p.output c.T (List.range (c.B*c.H))

theorem complete (c : Shape) (p : Storage.Placement c) (cs : List Config) (s : BlockState) :
    ∃ t, Host.sourceLaunches (source c p) cs s=some t ∧ Frame c p s t := by
  induction cs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons cfg rest ih =>
    obtain ⟨mid,he,_,hf⟩ := RealGrid.complete p.input p.output c.T (c.B*c.H) cfg p.distinct (List.range (c.B*c.H)) s
    obtain ⟨t,ht,hf'⟩ := ih mid
    refine ⟨t,?_,fun rg off hn => (hf' rg off hn).trans (hf rg off hn)⟩
    change (RealGrid.run p.input p.output c.T (c.B*c.H) cfg _ s).bind _=some t
    rw [he]
    exact ht

theorem tuned (c : Shape) (p : Storage.Placement c) (trials : List Config) (selected : Config) (s : BlockState) :
    ∃ t, Host.sourceLaunches (source c p) (trials++[selected]) s=some t ∧
      Frame c p s t ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        t.readMem p.output (row*c.T+time)=RealGrid.expected p.input c.T row time s) := by
  obtain ⟨mid,he,hf⟩ := complete c p trials s
  obtain ⟨t,ht,out,hf'⟩ := RealGrid.complete p.input p.output c.T (c.B*c.H) selected p.distinct (List.range (c.B*c.H)) mid
  refine ⟨t,?_,fun rg off hn => (hf' rg off hn).trans (hf rg off hn),?_⟩
  · rw [Host.sourceLaunches_append,he]
    simpa only [Host.sourceLaunches,source,ht,Option.bind_some]
  · intro row time hr ht'
    have inputs : mid.mem p.input=s.mem p.input := funext (fun off => hf p.input off (Or.inl (Ne.symm p.distinct)))
    simpa only [RealGrid.expected,BlockState.readMem,inputs] using out row (by simpa using hr) time ht'

def expected (logical : Nat → Nat → ℝ) (T row time : Nat) : ℝ :=
  ∑ k ∈ (Finset.range T).filter (fun k => k≤time), logical row k

theorem physical (c : Shape) (p : Storage.Placement c) (padding : RegionName → Nat → ℝ)
    (trials : List Config) (selected : Config) (heap : Nat → MemCell) :
    ∃ t, Host.sourceLaunches (source c p) (trials++[selected]) (entry c p padding heap)=some t ∧
      (∀ row time, row<c.B*c.H → time<c.T →
        read (Storage.commit c p heap t.mem (Storage.address p.backing.output (row*c.T+time)))=
          expected (fun row time => read (heap (Storage.address p.backing.input (row*c.T+time)))) c.T row time) ∧
      (∀ addr, ¬Storage.writes c p addr → Storage.commit c p heap t.mem addr=heap addr) := by
  obtain ⟨t,he,hf,out⟩ := tuned c p trials selected (entry c p padding heap)
  refine ⟨t,he,?_,?_⟩
  · intro row time hr ht
    rw [Storage.commit_output c p heap t.mem _ ((Storage.offset_bound c row time hr ht).trans_le p.backing.outputCells)]
    change t.readMem p.output (row*c.T+time)=_
    rw [out row time hr ht]
    simp only [expected,RealGrid.expected,BlockState.readMem,entry,Storage.lift_input,read]
    rfl
  · apply Storage.commit_frame c p (MemCell.real 0) heap t.mem
    intro off hn
    exact hf p.output off (Or.inr (fun row hr time ht => hn row (by simpa using hr) time ht))

#axiomsClean tuned
#axiomsClean physical
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.WholeReal
