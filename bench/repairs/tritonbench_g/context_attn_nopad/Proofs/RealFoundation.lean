import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.RealData

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000

private def tensorCapacity (t : Tensor3) := t.rows*t.rowStride+t.heads*t.headStride+t.dims*t.dimStride+1
private def commonCapacity (c : Call) := c.batches+tensorCapacity c.q+tensorCapacity c.k+tensorCapacity c.v+tensorCapacity c.o

private theorem tensor_bounded (t : Tensor3) : t.Bounded (tensorCapacity t) := by
  intro row head dim hr hh hd
  have a := Nat.mul_le_mul_right t.rowStride (Nat.le_of_lt hr)
  have b := Nat.mul_le_mul_right t.headStride (Nat.le_of_lt hh)
  have d := Nat.mul_le_mul_right t.dimStride (Nat.le_of_lt hd)
  dsimp [Tensor3.offset,tensorCapacity]
  omega

/-- A finite arithmetic bound witness solely for reusing the old shape and
formula lemmas. No flat execution or allocation/disjointness claim uses it. -/
private def boundWitness (c : Call) : FlatAlloc := ⟨"unused",[],fun _ => 0,fun _ => commonCapacity c⟩

theorem memory_witness (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) : Host.Memory c r (boundWitness c) s := by
  refine ⟨shape,hmeta.start,hmeta.length,?_,?_,?_,?_,?_,?_,labels.1,labels.2.1,labels.2.2.1,labels.2.2.2.1,labels.2.2.2.2⟩
  · dsimp [boundWitness,commonCapacity]; omega
  · dsimp [boundWitness,commonCapacity]; omega
  · intro row head dim hr hh hd
    exact lt_of_lt_of_le (tensor_bounded c.q row head dim hr hh hd) (by dsimp [boundWitness,commonCapacity]; omega)
  · intro row head dim hr hh hd
    exact lt_of_lt_of_le (tensor_bounded c.k row head dim hr hh hd) (by dsimp [boundWitness,commonCapacity]; omega)
  · intro row head dim hr hh hd
    exact lt_of_lt_of_le (tensor_bounded c.v row head dim hr hh hd) (by dsimp [boundWitness,commonCapacity]; omega)
  · intro row head dim hr hh hd
    exact lt_of_lt_of_le (tensor_bounded c.o row head dim hr hh hd) (by dsimp [boundWitness,commonCapacity]; omega)

def programWrites (c : Call) (r : Regions) (s : BlockState) (idx : GridIndex c.grid) : WriteFootprint :=
  fun cell => ∃ lane : RealIO.Lane c.layout,
    RealIO.active c.layout (c.launch idx) (Program.inputs s r) lane ∧
    cell.1=r.out ∧ cell.2=RealIO.write c.layout (c.launch idx) (Program.inputs s r) lane

theorem writes_iff (c : Call) (r : Regions) (s : BlockState) (idx : GridIndex c.grid)
    (region : RegionName) (off : Nat) :
    programWrites c r s idx (region,off) ↔
      EpilogueResult.Writes c.layout r (s.withGridIndex idx) region off := by
  simp only [programWrites,EpilogueResult.Writes,EpilogueResult.activeLane,EpilogueResult.offset,
    launch_grid,inputs_grid,RealIO.active,RealIO.write,AddressBounds.outputAddress,
    EpilogueExecution.address,BlockState.withGridIndex,BlockState.withNumPids,BlockState.withPids]
  constructor <;> rintro ⟨i,hi,hr,ho⟩ <;> exact ⟨i,hi,hr.symm,ho.symm⟩

theorem program_result (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r)
    (s : BlockState) (hmeta : Metadata c r s) (idx : GridIndex c.grid) :
    ∃ t,exec (kernel c r).eraseDType.toAlgKernel (s.withGridIndex idx)=some t ∧
      (∀ lane : RealIO.Lane c.layout,RealIO.active c.layout (c.launch idx) (Program.inputs s r) lane →
        t.readMem r.out (RealIO.write c.layout (c.launch idx) (Program.inputs s r) lane)=
          c.expected (Program.inputs s r) (c.launch idx).batch (c.launch idx).head
            (Formula.queryRow c.layout (c.launch idx) lane.1.val) lane.2.1.val) ∧
      BlockState.WriteWithin (programWrites c r s idx) (s.withGridIndex idx) t := by
  have hm := memory_witness c r shape labels s hmeta
  have hstart : 0 ≤ (Program.inputs (s.withGridIndex idx) r).start (Program.launch (s.withGridIndex idx)).batch := by
    rw [launch_grid,inputs_grid,hmeta.start _ (launch_batch_lt c idx)]
    exact Int.natCast_nonneg _
  have valid : RealIO.valid c.layout (Program.launch (s.withGridIndex idx))
      (Program.inputs (s.withGridIndex idx) r) := by
    rw [launch_grid,inputs_grid]
    exact Host.valid hm idx
  obtain ⟨t,run,value,frame⟩ := EpilogueResult.correct c.layout r .fp32 Bool.false c.scale
    (s.withGridIndex idx) (by norm_num [Call.layout]) hstart valid
  refine ⟨t,run,?_,?_⟩
  · intro lane on
    have result := value lane on
    change t.readMem r.out (RealIO.write c.layout (c.launch idx) (Program.inputs s r) lane)=
      RealIO.expected c.layout c.scale (c.launch idx) (Program.inputs s r) lane at result
    rw [RealIO.expected_closed,Host.closedExpected_logical hm idx lane on] at result
    exact result
  · intro region off outside
    exact (frame region off (fun hit => outside ((writes_iff c r s idx region off).mpr hit))).symm

#axiomsClean program_result
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
