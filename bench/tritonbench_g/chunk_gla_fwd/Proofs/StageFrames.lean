import bench.tritonbench_g.chunk_gla_fwd.Proofs.IntraTail
import bench.tritonbench_g.chunk_gla_fwd.Proofs.SplitMerge
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.StageFrames
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem inter_region (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (hr : r≠"attention") (off : Nat) :
    (InterGrid.result c scale s hu).mem r off=s.mem r off := by
  apply InterGrid.unwritten
  rintro q ⟨hc,_⟩
  exact hr hc

theorem intra_region (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (r : RegionName) (hr : r≠"attention") (off : Nat) :
    (IntraGrid.result c BK hK scale s hu).mem r off=s.mem r off := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hq⟩
  exact hr (IntraGrid.footprint_tile c BK q _ hq).1

theorem split_region (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (hr : r≠"scratch") (off : Nat) :
    (SplitGrid.result c scale s hu).mem r off=s.mem r off := by
  apply SplitGrid.unwritten
  rintro q ⟨_,hc,_⟩
  exact hr hc

theorem merge_region (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (hr : r≠"attention") (off : Nat) :
    (MergeGrid.result c s hu).mem r off=s.mem r off := by
  apply MergeGrid.unwritten
  rintro q ⟨hc,_⟩
  exact hr hc

theorem inputs_of_frame (s t : BlockState)
    (hf : ∀ (r : RegionName), r≠"attention" → r≠"scratch" → ∀ off, t.mem r off=s.mem r off) :
    InterGrid.inputs t=InterGrid.inputs s := by
  unfold InterGrid.inputs
  congr 1 <;> funext off <;> unfold BlockState.readMem
  · rw [hf "q" (by decide) (by decide)]
  · rw [hf "k" (by decide) (by decide)]
  · rw [hf "g" (by decide) (by decide)]
  · rw [hf "v" (by decide) (by decide)]
  · rw [hf "h" (by decide) (by decide)]

theorem inter_inputs (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (InterGrid.result c scale s hu)=InterGrid.inputs s :=
  inputs_of_frame s _ (fun r hr _ => inter_region c scale s hu r hr)

theorem intra_inputs (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (IntraGrid.result c BK hK scale s hu)=InterGrid.inputs s :=
  inputs_of_frame s _ (fun r hr _ => intra_region c BK hK scale s hu r hr)

theorem split_inputs (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (SplitGrid.result c scale s hu)=InterGrid.inputs s :=
  inputs_of_frame s _ (fun r _ hr => split_region c scale s hu r hr)

theorem merge_inputs (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (MergeGrid.result c s hu)=InterGrid.inputs s :=
  inputs_of_frame s _ (fun r hr _ => merge_region c s hu r hr)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.StageFrames
