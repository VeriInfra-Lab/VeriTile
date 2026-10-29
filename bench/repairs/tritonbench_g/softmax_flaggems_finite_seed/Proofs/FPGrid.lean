import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPGraph
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridMetadata

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGrid
open VeriTile Triton FP.Structural FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

def forwardKernel (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) : ComputeKernel :=
  if 1<d.K then Source.forward_noninner Y X d.M d.N d.K c.B c.C c.one
  else Source.forward_inner Y X d.M d.N c.B c.one

def backwardKernel (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) : ComputeKernel :=
  if 1<d.K then Source.softmax_backward_kernel_non_inner_fp32 P G Y d.M d.N d.K c.B c.C c.one
  else Source.softmax_backward_kernel_inner_fp32 P G Y d.M d.N c.C c.B c.one

def forwardBlock (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s : State α) : State α :=
  if 1<d.K then
    if c.one then scatter s Y (noninnerOffset row col d.N d.K) (noninnerActive col d.N d.K)
      (forwardNoninnerValue M s X row col d.N d.K c.B c.C)
    else FPForwardNoninner.graph M s X Y row col d.N d.K c.B c.C
  else
    if c.one then scatter s Y (innerOffset row d.N) (innerActive d.N)
      (forwardInnerValue M s X row d.N c.B)
    else FPForwardInner.graph M s X Y row d.N c.B

def backwardBlock (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s : State α) : State α :=
  if 1<d.K then
    if c.one then scatter s Y (noninnerOffset row col d.N d.K) (noninnerActive col d.N d.K)
      (backwardNoninnerValue M s P G row col d.N d.K c.B c.C)
    else FPBackwardNoninner.graph M s P G Y row col d.M d.N d.K c.B c.C
  else
    if c.one then scatter s Y (backwardInnerOffset row d.N) (backwardInnerActive row d.M d.N)
      (backwardInnerValue M s P G row d.M d.N c.B c.C)
    else FPBackwardInner.graph M s P G Y row d.M d.N c.B c.C

theorem forward_complete (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hB : 0<c.B) (s : State α) :
    ∃ t, exec M (forwardKernel X Y d c) s=some t ∧
      t.mem=(forwardBlock M X Y d c (s.pids 0) (s.pids 1) s).mem := by
  unfold forwardKernel forwardBlock
  split <;> cases hc : c.one
  · simpa using FPForwardNoninner.complete M s X Y d.M d.N d.K c.B c.C hB
  · simpa using FPOneTile.forward_noninner M s X Y d.M d.N d.K c.B c.C hB
  · simpa using FPForwardInner.complete M s X Y d.M d.N c.B hB
  · simpa using FPOneTile.forward_inner M s X Y d.M d.N c.B hB

theorem backward_complete (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s : State α) :
    ∃ t, exec M (backwardKernel P G Y d c) s=some t ∧
      t.mem=(backwardBlock M P G Y d c (s.pids 0) (s.pids 1) s).mem := by
  unfold backwardKernel backwardBlock
  split <;> cases hc : c.one
  · simpa using FPBackwardNoninner.complete M s P G Y d.M d.N d.K c.B c.C
  · simpa using FPOneTile.backward_noninner M s P G Y d.M d.N d.K c.B c.C
  · simpa using FPBackwardInner.complete M s P G Y d.M d.N c.B c.C
  · simpa using FPOneTile.backward_inner M s P G Y d.M d.N c.B c.C

theorem forward_congr (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s t : State α) (h : s.mem=t.mem) :
    (forwardBlock M X Y d c row col s).mem=(forwardBlock M X Y d c row col t).mem := by
  unfold forwardBlock
  split <;> cases hc : c.one
  · exact FPGraph.forward_noninner_congr M X Y row col d.N d.K c.B c.C s t h
  · apply FPGraphMemory.scatter_congr _ _ _ _ _ _ _ h
    delta forwardNoninnerValue forwardNoninnerExp forwardNoninnerLoad
    simp only [FPReference.read,h]
  · exact FPGraph.forward_inner_congr M X Y row d.N c.B s t h
  · apply FPGraphMemory.scatter_congr _ _ _ _ _ _ _ h
    delta forwardInnerValue forwardInnerExp forwardInnerLoad
    simp only [FPReference.read,h]

theorem backward_congr (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (row col : Nat) (s t : State α) (h : s.mem=t.mem) :
    (backwardBlock M P G Y d c row col s).mem=(backwardBlock M P G Y d c row col t).mem := by
  unfold backwardBlock
  split <;> cases hc : c.one
  · exact FPGraph.backward_noninner_congr M P G Y row col d.M d.N d.K c.B c.C s t h
  · apply FPGraphMemory.scatter_congr _ _ _ _ _ _ _ h
    delta backwardNoninnerValue backwardNoninnerScale backwardNoninnerLoad
    simp only [FPReference.read,h]
  · exact FPGraph.backward_inner_congr M P G Y row d.M d.N c.B c.C s t h
  · apply FPGraphMemory.scatter_congr _ _ _ _ _ _ _ h
    delta backwardInnerValue backwardInnerScale backwardInnerLoad
    simp only [FPReference.read,h]

def setup (gx gy : Nat) (p : Program) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then p.1 else if a=1 then p.2 else 0
          numPids := fun a => if a=0 then gx else if a=1 then gy else 1}

def run (M : Algebra α) (kernel : ComputeKernel) (gx gy : Nat) : List Program → State α → Option (State α)
  | [],s => some s
  | p::ps,s => (exec M kernel (setup gx gy p s)).bind (run M kernel gx gy ps)

def graphRun (block : Nat → Nat → State α → State α) : List Program → State α → State α
  | [],s => s
  | p::ps,s => graphRun block ps (block p.1 p.2 s)

theorem run_model (M : Algebra α) (kernel : ComputeKernel) (block : Nat → Nat → State α → State α)
    (total : ∀ s, ∃ t, exec M kernel s=some t ∧ t.mem=(block (s.pids 0) (s.pids 1) s).mem)
    (congr : ∀ row col s t, s.mem=t.mem → (block row col s).mem=(block row col t).mem)
    (gx gy : Nat) (ps : List Program) (s ref : State α) (h : s.mem=ref.mem) :
    ∃ t, run M kernel gx gy ps s=some t ∧ t.mem=(graphRun block ps ref).mem := by
  induction ps generalizing s ref with
  | nil => exact ⟨s,rfl,h⟩
  | cons p ps ih =>
    obtain ⟨mid,he,hm⟩ := total (setup gx gy p s)
    have same : mid.mem=(block p.1 p.2 ref).mem := by
      simpa [setup] using hm.trans (congr p.1 p.2 (setup gx gy p s) ref h)
    obtain ⟨final,hf,hv⟩ := ih mid (block p.1 p.2 ref) same
    exact ⟨final,by simp [run,he,hf],hv⟩

def forwardGrid (d : Dimensions) (c : LaunchConfig) : Nat × Nat :=
  if 1<d.K then (d.M,RealAddress.rounds d.K c.C) else (d.M,1)
def backwardGrid (d : Dimensions) (c : LaunchConfig) : Nat × Nat :=
  if 1<d.K then (d.M,RealAddress.rounds d.K c.C) else (RealAddress.rounds d.M c.C,1)

def forwardRun (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := forwardGrid d c
  run M (forwardKernel X Y d c) g.1 g.2 (grid g.1 g.2) s

def backwardRun (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := backwardGrid d c
  run M (backwardKernel P G Y d c) g.1 g.2 (grid g.1 g.2) s

def forwardGraph (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := forwardGrid d c
  graphRun (forwardBlock M X Y d c) (grid g.1 g.2) s

def backwardGraph (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := backwardGrid d c
  graphRun (backwardBlock M P G Y d c) (grid g.1 g.2) s

theorem forward_whole (M : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hB : 0<c.B) (s ref : State α) (h : s.mem=ref.mem) :
    ∃ t, forwardRun M X Y d c s=some t ∧ t.mem=(forwardGraph M X Y d c ref).mem :=
  run_model M (forwardKernel X Y d c) (forwardBlock M X Y d c)
    (forward_complete M X Y d c hB) (forward_congr M X Y d c)
    _ _ _ s ref h

theorem backward_whole (M : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s ref : State α) (h : s.mem=ref.mem) :
    ∃ t, backwardRun M P G Y d c s=some t ∧ t.mem=(backwardGraph M P G Y d c ref).mem :=
  run_model M (backwardKernel P G Y d c) (backwardBlock M P G Y d c)
    (backward_complete M P G Y d c) (backward_congr M P G Y d c)
    _ _ _ s ref h

#axiomsClean forward_whole
#axiomsClean backward_whole
#specNonCircular forwardGraph avoiding [forwardRun,run,FP.Structural.exec,forwardKernel]
#specNonCircular backwardGraph avoiding [backwardRun,run,FP.Structural.exec,backwardKernel]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGrid
