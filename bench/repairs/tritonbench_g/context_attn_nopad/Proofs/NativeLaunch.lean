import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeProgram
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WholeHost

namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeLaunch
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open HostPlan
noncomputable section
set_option maxHeartbeats 2000000

/-- One successful actual chunk/local-program execution, with the complete
finite index projection and its ordinary memory frame. -/
structure Frame (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) (j : Job c) where
  final : BlockState
  writes : WriteFootprint
  projection : NativeProgram.Projects c r A s j
  execution : NativeProgram.execute c r A j (A.flattenState s) = some final
  frame : BlockState.WriteWithin writes (A.flattenState s) final

abbrev Frames (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) :=
  (j : Job c) → Frame c r A s j

def Disjoint {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (fs : Frames c r A s) : Prop :=
  ∀ i j, i ≠ j → WriteFootprint.disjoint (fs i).writes (fs j).writes

/-- Merge of the actual source's finite chunk family. This is the same ordinary
disjoint-launch semantics used by the repository's full-grid model, now indexed
by native chunk bases and local PIDs. -/
def merge {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (fs : Frames c r A s) : BlockState := by
  classical
  exact
  { (A.flattenState s) with mem := fun rg off =>
      if h : (∃ j : Job c, (fs j).writes (rg,off)) then
        (fs (Classical.choose h)).final.mem rg off else (A.flattenState s).mem rg off }

structure Launched (c : Call) (r : Regions) (A : FlatAlloc) (s final : BlockState) where
  frames : Frames c r A s
  disjoint : Disjoint frames
  final_eq : final = merge frames

def ofGridFrames {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hp : ∀ j : Job c, NativeProgram.Projects c r A s j)
    (fs : Kernel.GridFrames (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
      c.grid (A.flattenState s)) : Frames c r A s := fun j => {
  final := (fs (index c j)).final
  writes := (fs (index c j)).writes
  projection := hp j
  execution := by rw [NativeProgram.execute_projects (hp j)]; exact (fs (index c j)).h_exec
  frame := (fs (index c j)).h_writeWithin }

theorem ofGrid_disjoint {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hp : ∀ j : Job c, NativeProgram.Projects c r A s j)
    (fs : Kernel.GridFrames (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
      c.grid (A.flattenState s)) (hd : Kernel.GridWritesDisjoint fs) :
    Disjoint (ofGridFrames hp fs) := by
  intro i j hne
  exact hd _ _ (fun he => hne ((jobEquiv c).injective he))

theorem merge_written {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (fs : Frames c r A s) (hd : Disjoint fs) (j : Job c) (rg : RegionName) (off : Nat)
    (hw : (fs j).writes (rg,off)) : (merge fs).mem rg off = (fs j).final.mem rg off := by
  classical
  unfold merge
  have he : ∃ i : Job c, (fs i).writes (rg,off) := ⟨j,hw⟩
  dsimp only
  rw [dif_pos he]
  have hj : Classical.choose he = j := by
    by_contra hne
    exact hd _ _ hne _ (Classical.choose_spec he) hw
  rw [hj]

theorem merge_unwritten {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (fs : Frames c r A s) (rg : RegionName) (off : Nat)
    (hw : ¬ ∃ j : Job c, (fs j).writes (rg,off)) :
    (merge fs).mem rg off = (A.flattenState s).mem rg off := by
  classical
  unfold merge
  dsimp only
  rw [dif_neg hw]

theorem merge_ofGrid {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hp : ∀ j : Job c, NativeProgram.Projects c r A s j)
    (fs : Kernel.GridFrames (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
      c.grid (A.flattenState s)) (hd : Kernel.GridWritesDisjoint fs) :
    merge (ofGridFrames hp fs) = Kernel.mergeFrames c.grid (A.flattenState s) fs := by
  have hmem : (merge (ofGridFrames hp fs)).mem =
      (Kernel.mergeFrames c.grid (A.flattenState s) fs).mem := by
    funext rg off
    by_cases hw : ∃ idx : GridIndex c.grid, (fs idx).writes (rg,off)
    · obtain ⟨idx, hidx⟩ := hw
      have hj : (ofGridFrames hp fs (splitIndex c idx)).writes (rg,off) := by
        change (fs (HostPlan.index c (HostPlan.splitIndex c idx))).writes (rg,off)
        rw [HostPlan.index_splitIndex]
        exact hidx
      rw [merge_written _ (ofGrid_disjoint hp fs hd) (splitIndex c idx) rg off hj,
        Kernel.mergeFrames_mem_written hd idx rg off hidx]
      change (fs (HostPlan.index c (HostPlan.splitIndex c idx))).final.mem rg off = _
      rw [HostPlan.index_splitIndex]
    · rw [merge_unwritten, Kernel.mergeFrames_mem_unwritten rg off hw]
      rintro ⟨j,hj⟩
      exact hw ⟨index c j,hj⟩
  exact congrArg (fun memory => { (A.flattenState s) with mem := memory }) hmem

/-- Total source-program bridge from the original proven logical grid to the
actual widened/partitioned entry. Neither execution success nor output equality
is an assumption: the old grid witness is obtained by the reused correctness
proof and each native finite-index projection is proved separately. -/
def from_grid {c : Call} {r : Regions} {A : FlatAlloc} {s final : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hb : c.blocks ≤ 16777216)
    (hg : Kernel.GridLaunchedOrdinary
      (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
      c.grid (A.flattenState s) final) : Launched c r A s final := by
  let hp := NativeProgram.projects hm ha hb
  exact ⟨ofGridFrames hp hg.frames, ofGrid_disjoint hp hg.frames hg.h_disjoint,
    hg.h_final.trans (merge_ofGrid hp hg.frames hg.h_disjoint).symm⟩

/-- Public full host trace: exact max/item synchronization, unchanged state at
that read, every split-axis native program, one merged finite output, None. -/
def Returns (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (result : BlockState × Option Unit) : Prop :=
  ∃ synchronizedMaximum : Nat, synchronizedMaximum = metadataMaximum c r s ∧
    Nonempty (Launched (NativeIndex.capCall c synchronizedMaximum) r A s result.1) ∧
    result.2 = none

theorem from_logical_returns {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A)
    {result : BlockState × Option Unit}
    (hr : (effective c r s).Returns r A s Bool.false result) : Returns c r A s result := by
  refine ⟨metadataMaximum c r s, rfl, ?_, hr.2⟩
  obtain ⟨hg⟩ := hr.1
  exact ⟨from_grid (effective_memory hm) ⟨ha.heads,ha.bytes⟩ (NativeProgram.effective_blocks_bound hm) hg⟩

end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeLaunch
