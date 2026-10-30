import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealContract
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSafety
import VeriTile.Triton.Memory.FlattenFrame

/-! Finite-allocation contract for the complete persistent real projection.
Inputs are the original tagged real A/B cells, not finite-fallback observations
or prepared dot accumulators. Output windows contain exactly the active cells
assigned by the source schedule to one legal host-grid PID. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000

structure Admissible (p : Parameters) : Prop where
  m : 0 < p.M
  n : 0 < p.N
  k : 0 < p.K
  bm : 0 < p.BM
  bn : 0 < p.BN
  bk : 0 < p.BK
  gm : 0 < p.GM
  sms : 0 < p.SMS
  output_a : p.C ≠ p.A
  output_b : p.C ≠ p.B
  output_injective : RealOutputAddresses.Injective p

/-- The Python wrapper launches min(NUM_SMS, ceil(M/BM)*ceil(N/BN)). -/
def grid (p : Parameters) : Nat :=
  min p.SMS (RealGroup.rows p * RealGroup.columns p)

structure Launch (p : Parameters) (s : BlockState) : Prop where
  grid_x : s.numPids 0 = grid p
  pid_x : s.pids 0 < grid p
  grid_y : s.numPids 1 = 1
  grid_z : s.numPids 2 = 1
  pid_y : s.pids 1 = 0
  pid_z : s.pids 2 = 0

theorem Launch.pid_bound {p : Parameters} {s : BlockState} (h : Launch p s) :
    s.pids 0 < p.SMS := lt_of_lt_of_le h.pid_x (Nat.min_le_left _ _)

def outputWindow (p : Parameters) (pid q : Nat) (j : TileIndex [p.BM,p.BN]) : Nat :=
  RealStore.offset p (RealLoop.tileRow p pid q) (RealLoop.tileColumn p pid q) j

def outputMask (p : Parameters) (pid q : Nat) (j : TileIndex [p.BM,p.BN]) : Prop :=
  q < RealLoopState.work p pid ∧
    RealStore.active p (RealLoop.tileRow p pid q) (RealLoop.tileColumn p pid q) j

def expected (p : Parameters) (xs : RealMath.Inputs) (pid q : Nat)
    (j : TileIndex [p.BM,p.BN]) : ℝ :=
  RealMath.formula p.K xs
    (RealStore.rows p (RealLoop.tileRow p pid q) j.1)
    (RealStore.columns p (RealLoop.tileColumn p pid q) j.2.1)

def Written (p : Parameters) (pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  ∃ q j, outputMask p pid q j ∧ r = p.C ∧ o = outputWindow p pid q j

/-- The frame excludes only active source output addresses; allocated stride
gaps, all other buffers, masked lanes and unallocated flat cells are preserved. -/
def Outside (p : Parameters) (A : FlatAlloc) (pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ A.flat ∨ ∀ q j, outputMask p pid q j → o ≠ A.addr p.C (outputWindow p pid q j)

structure Results (p : Parameters) (A : FlatAlloc) (xs : RealMath.Inputs)
    (s t : BlockState) : Prop where
  output : ∀ q j, outputMask p (s.pids 0) q j →
    t.mem A.flat (A.addr p.C (outputWindow p (s.pids 0) q j)) =
      MemCell.real (expected p xs (s.pids 0) q j)
  input_a : ∀ r k, r < p.M → k < p.K →
    t.mem A.flat (A.addr p.A (r*p.sam+k*p.sak)) = MemCell.real (xs.a r k)
  input_b : ∀ k c, k < p.K → c < p.N →
    t.mem A.flat (A.addr p.B (k*p.sbk+c*p.sbn)) = MemCell.real (xs.b k c)
  frame : ∀ r o, Outside p A (s.pids 0) r o →
    t.mem r o = (A.flattenState s).mem r o

/-- A successful full-source run in one finite flat address space, independent
GEMM results on every assigned active output, original inputs and full frame.
The undef condition is the shared finite-allocation bridge's convention. -/
def Implements (p : Parameters) (kernel : Kernel) : Prop :=
  ∀ (A : FlatAlloc) (s : BlockState) (xs : RealMath.Inputs),
    A.Disjoint → A.regions = [p.A,p.B,p.C] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
    s.undef = (fun _ _ => 0) → Launch p s →
    RealAccess.Bounds p A.extent → RealInput.Represents p xs s →
    ∃ t, exec (A.flattenKernel kernel) (A.flattenState s) = some t ∧ Results p A xs s t

theorem results (p : Parameters) (A : FlatAlloc) (xs : RealMath.Inputs) (s t : BlockState)
    (hd : A.Disjoint) (hr : A.regions = [p.A,p.B,p.C])
    (hb : RealAccess.Bounds p A.extent)
    (hy : RealCompleted.Holds p xs (s.pids 0) (RealLoopState.work p (s.pids 0)) t)
    (hi : RealInput.Represents p xs t)
    (hf : ∀ r o, RealLoop.Outside p (s.pids 0) r o → t.mem r o = s.mem r o) :
    Results p A xs s (A.flattenState t) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro q j hj
    have hm := A.flattenState_mem_addr hd t (r := p.C)
      (o := outputWindow p (s.pids 0) q j) (by simp [hr])
      (hb.c _ _ hj.2.1 hj.2.2)
    exact hm.trans (congrArg A.trCell (hy q hj.1 j hj.2))
  · intro r k hrow hk
    rw [A.flattenState_mem_addr hd t (by simp [hr]) (hb.a r k hrow hk),hi.a r k hrow hk]
    rfl
  · intro k c hk hc
    rw [A.flattenState_mem_addr hd t (by simp [hr]) (hb.b k c hk hc),hi.b k c hk hc]
    rfl
  · intro r o hout
    apply A.flattenState_frame s t (Written p (s.pids 0))
    · intro region offset hnot
      apply hf region offset
      intro q hq j hj he
      exact hnot ⟨q,j,⟨hq,hj⟩,he.1.symm,he.2.symm⟩
    · rcases hout with hout | hout
      · exact Or.inl hout
      · refine Or.inr ?_
        intro region offset hw
        obtain ⟨q,j,hj,rfl,rfl⟩ := hw
        exact hout q j hj

theorem original_correct (p : Parameters) (h : Admissible p) :
    Implements p (RealSource.kernel p Bool.false) := by
  intro A s xs hd hr hc hu hl hb hi
  obtain ⟨t,he,hy,hinput,hframe⟩ := RealContract.executes p xs s
    h.m h.n h.k h.bm h.bn h.bk h.gm h.sms hl.pid_bound h.output_a h.output_b h.output_injective hi
  have hsafe := RealSafety.safe p A.extent xs s h.m h.n h.k h.bk h.gm h.sms
    hl.pid_bound h.output_a h.output_b hb hi
  refine ⟨A.flattenState t,?_,results p A xs s t hd hr hb hy hinput hframe⟩
  rw [A.exec_flatten hd hc _ s hsafe (RealPureSafety.flatten_ok p Bool.false) hu]
  change (stepStmts (RealSource.kernel p Bool.false).body s).map A.flattenState = _
  rw [he,Option.map_some]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIO
