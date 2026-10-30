import bench.tritonbench_g.chunk_gated_attention.Sources.HostProgram
import bench.tritonbench_g.chunk_gated_attention.Sources.Ownership

/-! One actual legal x-axis batch: every program has a constructed successful
finite-allocation execution; disjointness and merged logical outputs follow
from decoded ownership. Uses the shared ordinary-grid launch semantics. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Batch
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def grid (size : Nat) : VeriTile.Triton.Grid := ⟨[size]⟩
def lane {size : Nat} (i : GridIndex (grid size)) : Nat := (i ⟨0,by simp [grid,VeriTile.Triton.Grid.rank]⟩).val

theorem lane_lt {size : Nat} (i : GridIndex (grid size)) : lane i < size :=
  (i ⟨0,by simp [grid,VeriTile.Triton.Grid.rank]⟩).isLt

theorem pid_lane {size : Nat} (i : GridIndex (grid size)) (u : BlockState) :
    (u.withGridIndex i).pids 0 = lane i := by
  simp [BlockState.withGridIndex,BlockState.withNumPids,BlockState.withPids,GridIndex.toPids,grid,
    VeriTile.Triton.Grid.rank,lane]

theorem lane_injective {size : Nat} {i j : GridIndex (grid size)} (h : lane i = lane j) : i=j := by
  funext ax
  have hz : ax.val = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ax.isLt)
  have ha : ax = ⟨0,by simp [grid,VeriTile.Triton.Grid.rank]⟩ := Fin.ext hz
  subst ax
  exact Fin.ext h

theorem decoded_injective (p : Layout) (base size : Nat) (u : BlockState)
    (hx : 0 < p.V/p.BV) (hy : 0 < p.K/p.BK)
    (i j : GridIndex (grid size))
    (h : RebasedInit.decoded p base (u.withGridIndex i) = RebasedInit.decoded p base (u.withGridIndex j)) : i=j := by
  have he := congrArg (fun l : Launch => Grid.encode (p.V/p.BV) (p.K/p.BK) l.value l.key l.head) h
  change Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) =
    Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) at he
  rw [Grid.decode_encode_identity hx hy,Grid.decode_encode_identity hx hy,pid_lane,pid_lane] at he
  exact lane_injective (Nat.add_right_cancel he)

def writes (A : FlatAlloc) (p : Layout) (r : Regions) (base size : Nat) (storeFinal : Bool) (u : BlockState) : WriteFootprint :=
  fun cell => ∃ idx : GridIndex (grid size), RebasedFlat.footprint A p
    (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal cell

theorem executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (heads T K V : Nat) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base size : Nat) (gateK hasSeed storeFinal swapped : Bool)
    (x : Nat → Host.InnerInputs) (u : BlockState)
    (hp : StagedLayout.Canonical p) (hk : K ≤ p.K) (hv : V ≤ p.V)
    (hsize : 0 < size) (hgrid : size ≤ Grid.limit)
    (hbatch : base+size ≤ heads*(p.K/p.BK)*(p.V/p.BV))
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (ha : StagedLayout.Allocated p heads r gateK hasSeed storeFinal A.extent)
    (hkey : ∀ off, u.readMem r.key off = (Materialize.inputs p T K V gateK x).key off)
    (hvalue : ∀ off, u.readMem r.value off = (Materialize.inputs p T K V gateK x).value off)
    (hgate : ∀ off, u.readMem r.gate off = (Materialize.inputs p T K V gateK x).gate off)
    (hseed : ∀ off, u.readMem r.seed off = (Materialize.inputs p T K V gateK x).seed off) :
    ∃ v, ∃ _ : Kernel.GridLaunchedOrdinary
      (A.flattenKernel (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel)
      (grid size) (A.flattenState u) v,
      (∀ idx : GridIndex (grid size),
        HostProgram.Outputs A p (RebasedInit.decoded p base (u.withGridIndex idx)) r
          (x (RebasedInit.decoded p base (u.withGridIndex idx)).head) T K V gateK hasSeed storeFinal v) ∧
      BlockState.WriteWithin (writes A p r base size storeFinal u) (A.flattenState u) v := by
  classical
  have hpid (idx : GridIndex (grid size)) :
      (u.withGridIndex idx).pids 0+base < heads*(p.K/p.BK)*(p.V/p.BV) := by
    rw [pid_lane]
    have := lane_lt idx
    omega
  have hex (idx : GridIndex (grid size)) := HostProgram.executes A hd hcov p heads T K V r dot gate history
    seed final base gateK hasSeed storeFinal swapped x (u.withGridIndex idx) hp hk hv (hpid idx)
    hs hfsep ha hkey hvalue hgate hseed
  choose out he ho hf using hex
  let kernel := A.flattenKernel (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel
  let frames : Kernel.GridFrames kernel (grid size) (A.flattenState u) := fun idx =>
    ⟨out idx,RebasedFlat.footprint A p (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal,he idx,hf idx⟩
  have ht : 0 < heads*(p.K/p.BK)*(p.V/p.BV) := by omega
  have hx : 0 < p.V/p.BV := by
    by_contra hn
    simp [Nat.eq_zero_of_not_pos hn] at ht
  have hy : 0 < p.K/p.BK := by
    by_contra hn
    simp [Nat.eq_zero_of_not_pos hn] at ht
  have hdis : Kernel.GridWritesDisjoint frames := by
    intro i j hne cell hi hj
    have heq := Ownership.flat_owner A hd hcov p heads _ _ r gateK hasSeed storeFinal hp
      (StagedLayout.decoded_valid p heads base (u.withGridIndex i) (hpid i))
      (StagedLayout.decoded_valid p heads base (u.withGridIndex j) (hpid j)) ha hfsep cell hi hj
    exact hne (decoded_injective p base size u hx hy i j heq)
  let v := Kernel.mergeFrames (grid size) (A.flattenState u) frames
  refine ⟨v,⟨frames,hdis,rfl⟩,?_,Kernel.mergeFrames_writeWithin frames⟩
  intro idx
  have hm (rg : RegionName) (off : Nat)
      (hw : RebasedFlat.footprint A p (RebasedInit.decoded p base (u.withGridIndex idx)) r storeFinal (rg,off)) :
      v.readMem rg off = (out idx).readMem rg off := by
    unfold BlockState.readMem
    rw [show v.mem rg off = (frames idx).final.mem rg off from Kernel.mergeFrames_mem_written hdis idx rg off hw]
  constructor
  · intro c e j hc hek hjv
    rw [hm _ _ (by
      refine ⟨rfl,r.history,_,Or.inl ⟨rfl,c,e,j,hc,⟨lt_of_lt_of_le hek hk,lt_of_lt_of_le hjv hv⟩,rfl⟩,rfl⟩)]
    exact (ho idx).history c e j hc hek hjv
  · intro hon e j hek hjv
    rw [hm _ _ (by
      refine ⟨rfl,r.final,_,Or.inr ⟨hon,rfl,e,j,⟨lt_of_lt_of_le hek hk,lt_of_lt_of_le hjv hv⟩,?_⟩,rfl⟩
      simp [RealHStore.address,RealHFinal.base,stateAddress])]
    exact (ho idx).final hon e j hek hjv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Batch
