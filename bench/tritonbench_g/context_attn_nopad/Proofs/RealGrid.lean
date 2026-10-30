import bench.tritonbench_g.context_attn_nopad.Proofs.RealFoundation
import bench.tritonbench_g.context_attn_nopad.Proofs.IOBounds

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000

private def frames (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) :
    Kernel.GridFrames (kernel c r).eraseDType.toAlgKernel c.grid s := fun idx => {
  final := Classical.choose (program_result c r shape labels s hmeta idx)
  writes := programWrites c r s idx
  h_exec := (Classical.choose_spec (program_result c r shape labels s hmeta idx)).1
  h_writeWithin := (Classical.choose_spec (program_result c r shape labels s hmeta idx)).2.2 }

private theorem frames_disjoint (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) :
    Kernel.GridWritesDisjoint (frames c r shape labels s hmeta) := by
  intro idx idx' hne cell hi hj
  rcases hi with ⟨i,hai,_,hei⟩
  rcases hj with ⟨j,haj,_,hej⟩
  exact hne (Host.unique_writer (memory_witness c r shape labels s hmeta)
    idx idx' i j hai haj (hei.symm.trans hej)).1

def writes (c : Call) (r : Regions) (s : BlockState) : WriteFootprint :=
  fun cell => ∃ idx : GridIndex c.grid,programWrites c r s idx cell

/-- Regional version of the original independent ordinary-grid result. Each
CTA still sees the original memory; no serial update semantics is substituted. -/
theorem grid_result (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) :
    ∃ final,Nonempty (Kernel.GridLaunchedOrdinary
        (kernel c r).eraseDType.toAlgKernel c.grid s final) ∧
      (∀ j : Coordinate c,IOData.active c j →
        final.readMem r.out (IOData.address c c.o j)=
          c.expected (Program.inputs s r) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val) ∧
      BlockState.WriteWithin (writes c r s) s final := by
  let fs := frames c r shape labels s hmeta
  have hd := frames_disjoint c r shape labels s hmeta
  refine ⟨Kernel.mergeFrames c.grid s fs,⟨⟨fs,hd,rfl⟩⟩,?_,Kernel.mergeFrames_writeWithin fs⟩
  intro j on
  have hm := memory_witness c r shape labels s hmeta
  obtain ⟨idx,lane,hb,hh,ht,hch,active⟩ := Host.covered hm
    j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
    j.1.isLt j.2.1.isLt on j.2.2.2.1.isLt j.2.2.1.isLt
  have addr : RealIO.write c.layout (c.launch idx) (Program.inputs s r) lane=
      IOData.address c c.o j := by
    rw [Host.write_eq hm,hb,hh,ht,hch]
    rfl
  have hw : (fs idx).writes (r.out,IOData.address c c.o j) := ⟨lane,active,rfl,addr.symm⟩
  have he := Kernel.mergeFrames_mem_written hd idx _ _ hw
  unfold BlockState.readMem
  rw [he,←addr]
  simpa only [frames,BlockState.readMem,hb,hh,ht,hch] using
    (Classical.choose_spec (program_result c r shape labels s hmeta idx)).2.1 lane active

theorem outside_writes (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) (xs : Values c) (region : RegionName) (off : Nat)
    (outside : ∀ j : Fin (laneCount c),IOData.active c (decode c j) →
      region≠r.out ∨ off≠IOData.address c c.o (decode c j)) :
    ¬writes c r s (region,off) := by
  rintro ⟨idx,lane,on,hr,hoff⟩
  have hm := memory_witness c r shape labels s hmeta
  let j := IOData.coordinate c idx _ (IOData.row_bound c idx lane.1) lane.2.1
  have active : IOData.active c j := (Host.active_iff hm idx lane).mp on
  have addr : RealIO.write c.layout (c.launch idx) (Program.inputs s r) lane=IOData.address c c.o j :=
    Host.write_eq hm idx lane
  have hout := outside (encode c j) (by simpa only [decode_encode] using active)
  simp only [decode_encode] at hout
  exact hout.elim (fun h=>h hr) (fun h=>h (hoff.trans addr))

#axiomsClean grid_result
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
