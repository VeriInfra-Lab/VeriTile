import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.CheckedHost
import bench.repairs.tritonbench_g.chunk_delta_fwd.RealOutputs

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.RealPhysical
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
noncomputable section
set_option maxHeartbeats 8000000

/-- Ideal Real observations use the same physical byte addresses. FP32 byte
payloads and their exact four-byte encoding are handled by ReturnedMemory. -/
abbrev Physical := Nat→ℝ

def input (r : Request) (a : Allocations r) (before : Physical) : Memory ℝ := fun region i=>
  .mk .real (match descriptor r a region with | some t=>before (t.byteBase+4*i) | none=>0)
def source (r : Request) (a : Allocations r) (before : Physical) :=
  gridMemory Real.algebra Real.dot r.params (grid r.params) (input r a before) (input r a before)

theorem source_complete (r : Request) (checked : r.check=Bool.true) (a : Allocations r) (before : Physical) :
    runGrid Real.algebra Real.dot r.params (grid r.params) (input r a before)=some (source r a before) ∧
    Real.Spec r.params (input r a before) (input r a before) (source r a before) :=
  ⟨(grid_execution _ _ _ _ _ _ (SameInputs.refl _)).1,
    Real.grid_to_real _ (r.shape checked) _ _ _ (grid_spec _ _ _ _ _ (List.Perm.refl _) _ _)⟩

def commit (before : Physical) (t : Tensor) (values : Nat→ℝ) : Physical := fun address=>
  if t.byteBase≤address ∧ address<t.byteBase+4*t.count then values ((address-t.byteBase)/4) else before address

theorem commit_read (before : Physical) (t : Tensor) (values : Nat→ℝ) (i : Nat) (hi : i<t.count) :
    commit before t values (t.byteBase+4*i)=values i := by
  simp [commit,show t.byteBase≤t.byteBase+4*i ∧ t.byteBase+4*i<t.byteBase+4*t.count by omega]
theorem commit_frame (before : Physical) (t : Tensor) (values : Nat→ℝ) (address : Nat) (outside : Bytes.Outside t address) :
    commit before t values address=before address := by
  simp [commit,show ¬(t.byteBase≤address ∧ address<t.byteBase+4*t.count) by unfold Bytes.Outside at outside;omega]
theorem commit_separate (before : Physical) (a b : Tensor) (values : Nat→ℝ)
    (separate : Separate a b) (i : Nat) (hi : i<b.count) : commit before a values (b.byteBase+4*i)=before (b.byteBase+4*i) := by
  rcases separate with ha|hb|hab|hba
  · simp [commit,ha]
  · omega
  · apply commit_frame;right;omega
  · apply commit_frame;left;omega

def publish (r : Request) (a : Allocations r) (before : Physical) (memory : Memory ℝ) : Physical :=
  let history:=commit before a.history (fun i=>(memory "history" i).read .real)
  let corrected:=commit history a.corrected (fun i=>(memory "corrected" i).read .real)
  match r.final with | none=>corrected | some f=>commit corrected f (fun i=>(memory "final" i).read .real)
def returned (r : Request) (a : Allocations r) (before : Physical) := publish r a before (source r a before)

/-- Independent mathematical outputs, pinned to the original descriptors.
The source recurrence keeps width ceil(BT/BC)*BC separately from stride BT. -/
def history (r : Request) (a : Allocations r) (before : Physical) (head chunk key value : Nat) :=
  Real.target r.params head (input r a before) chunk key value
def corrected (r : Request) (a : Allocations r) (before : Physical) (head time value : Nat) :=
  RealMath.corrected (Real.arrays r.params head (input r a before)) r.K
    (Real.target r.params head (input r a before) (time/r.BT)) time value

theorem history_read (r : Request) (checked : r.check=Bool.true) (a : Allocations r) (before : Physical)
    (head : Nat) (hh : head<r.B*r.H) (chunk : Nat) (hc : chunk<r.dims.NT)
    (key : Nat) (hk : key<r.K) (value : Nat) (hv : value<r.V) :
    returned r a before (a.history.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      history r a before head chunk key value := by
  have bound : ((head*r.dims.NT+chunk)*r.K+key)*r.V+value<a.history.count := by
    rw [a.history_count]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh hc) hk) hv
  have val:=(source_complete r checked a before).2.history head hh chunk hc key hk value hv
  change source r a before "history" (((head*r.dims.NT+chunk)*r.K+key)*r.V+value)=_ at val
  unfold returned publish
  cases hf : r.final with
  | none=>
    simp only
    rw [commit_separate _ a.corrected a.history _ (Bytes.Separate.symm a.pair) _ bound,
      commit_read _ _ _ _ bound,val,Cell.read_same]
    rfl
  | some f=>
    have sep:=(a.fresh a.history (by simp) f (by simp [Request.all,hf])).1
    simp only
    rw [commit_separate _ f a.history _ (Bytes.Separate.symm sep) _ bound,
      commit_separate _ a.corrected a.history _ (Bytes.Separate.symm a.pair) _ bound,
      commit_read _ _ _ _ bound,val,Cell.read_same]
    rfl

theorem corrected_read (r : Request) (checked : r.check=Bool.true) (a : Allocations r) (before : Physical)
    (head : Nat) (hh : head<r.B*r.H) (time : Nat) (ht : time<r.T) (value : Nat) (hv : value<r.V) :
    returned r a before (a.corrected.byteBase+4*((head*r.T+time)*r.V+value))=
      corrected r a before head time value := by
  have bound : (head*r.T+time)*r.V+value<a.corrected.count := by
    rw [a.corrected_count]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh ht) hv
  have val:=(source_complete r checked a before).2.corrected head hh time ht value hv
  change source r a before "corrected" ((head*r.T+time)*r.V+value)=_ at val
  unfold returned publish
  cases hf : r.final with
  | none=>simp only;rw [commit_read _ _ _ _ bound,val,Cell.read_same];rfl
  | some f=>
    have sep:=(a.fresh a.corrected (by simp) f (by simp [Request.all,hf])).1
    simp only
    rw [commit_separate _ f a.corrected _ (Bytes.Separate.symm sep) _ bound,
      commit_read _ _ _ _ bound,val,Cell.read_same]
    rfl

theorem final_read (r : Request) (checked : r.check=Bool.true) (a : Allocations r) (before : Physical)
    (f : Tensor) (hf : r.final=some f) (head : Nat) (hh : head<r.B*r.H)
    (key : Nat) (hk : key<r.K) (value : Nat) (hv : value<r.V) :
    returned r a before (f.byteBase+4*((head*r.K+key)*r.V+value))=history r a before head r.dims.NT key value := by
  have bound : (head*r.K+key)*r.V+value<f.count := by
    rw [r.state_count checked f (by simp [hf])]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh hk) hv
  have flag : r.params.storeFinal=Bool.true:=by simp [Request.params,Request.dims,Host.Dimensions.parameters,hf]
  have val:=(source_complete r checked a before).2.final flag head hh key hk value hv
  change source r a before "final" ((head*r.K+key)*r.V+value)=_ at val
  simp only [returned,publish,hf]
  rw [commit_read _ _ _ _ bound,val,Cell.read_same]
  rfl

theorem exterior (r : Request) (a : Allocations r) (before : Physical) (address : Nat)
    (history : Bytes.Outside a.history address) (corrected : Bytes.Outside a.corrected address)
    (final : ∀ f∈r.final.toList,Bytes.Outside f address) : returned r a before address=before address := by
  unfold returned publish
  cases hf : r.final with
  | none=>simp only;rw [commit_frame _ a.corrected _ address corrected,commit_frame _ a.history _ address history]
  | some f=>simp only;rw [commit_frame _ f _ address (final f (by simp [hf])),
      commit_frame _ a.corrected _ address corrected,commit_frame _ a.history _ address history]

#axiomsClean source_complete
#axiomsClean history_read
#axiomsClean corrected_read
#axiomsClean final_read
#axiomsClean exterior
#specNonCircular history avoiding [runGrid,gridMemory,source,states]
#specNonCircular corrected avoiding [runGrid,gridMemory,source,states]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.RealPhysical
