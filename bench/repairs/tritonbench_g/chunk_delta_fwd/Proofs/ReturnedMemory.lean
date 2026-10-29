import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.Storage

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
noncomputable section
set_option maxHeartbeats 8000000

/-- The full typed source is constructed on the same original byte inputs. -/
theorem source_complete (r : Request) (checked : r.check=Bool.true) (a : Allocations r) (before : Bytes.Memory) :
    runGrid Native.algebra Native.dot r.params (grid r.params) (input r a before)=some (source r a before) ∧
    GridSpec Native.algebra Native.dot r.params (r.shape checked) (input r a before) (input r a before) (source r a before) :=
  ⟨(grid_execution _ _ _ _ _ _ (SameInputs.refl _)).1,grid_spec _ _ _ _ _ (List.Perm.refl _) _ _⟩

theorem publish_history (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (memory : Memory Expr) (i : Nat) (hi : i<a.history.count) :
    Bytes.read (publish I r a before memory) (a.history.byteBase+4*i)=words I memory "history" i := by
  unfold publish
  cases hf : r.final with
  | none=>
    rw [Bytes.separate_read _ a.corrected a.history _ (Bytes.Separate.symm a.pair) i hi]
    exact Bytes.read_stores _ _ _ i hi
  | some f=>
    have mem : f∈r.all:=by simp [Request.all,hf]
    have sep:=(a.fresh a.history (by simp) f mem).1
    rw [Bytes.separate_read _ f a.history _ (Bytes.Separate.symm sep) i hi,
      Bytes.separate_read _ a.corrected a.history _ (Bytes.Separate.symm a.pair) i hi]
    exact Bytes.read_stores _ _ _ i hi

theorem publish_corrected (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (memory : Memory Expr) (i : Nat) (hi : i<a.corrected.count) :
    Bytes.read (publish I r a before memory) (a.corrected.byteBase+4*i)=words I memory "corrected" i := by
  unfold publish
  cases hf : r.final with
  | none=>exact Bytes.read_stores _ _ _ i hi
  | some f=>
    have mem : f∈r.all:=by simp [Request.all,hf]
    have sep:=(a.fresh a.corrected (by simp) f mem).1
    rw [Bytes.separate_read _ f a.corrected _ (Bytes.Separate.symm sep) i hi]
    exact Bytes.read_stores _ _ _ i hi

theorem publish_final (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (memory : Memory Expr) (f : Tensor) (hf : r.final=some f) (i : Nat) (hi : i<f.count) :
    Bytes.read (publish I r a before memory) (f.byteBase+4*i)=words I memory "final" i := by
  simp only [publish,hf]
  exact Bytes.read_stores _ _ _ i hi

theorem caller_frame (I : Primitives) (r : Request) (checked : r.check=Bool.true)
    (a : Allocations r) (before : Bytes.Memory) (memory : Memory Expr)
    (tensor : Tensor) (member : tensor∈r.inputs) (byte : Nat) (bound : byte<4*tensor.count) :
    publish I r a before memory (tensor.byteBase+byte)=before (tensor.byteBase+byte) := by
  have all : tensor∈r.all:=by simp [Request.all,member]
  have hsep:=(a.fresh a.history (by simp) tensor all).1
  have csep:=(a.fresh a.corrected (by simp) tensor all).1
  unfold publish
  cases hf : r.final with
  | none=>
    simp only
    rw [Bytes.separate_frame _ a.corrected tensor _ csep byte bound,
      Bytes.separate_frame _ a.history tensor _ hsep byte bound]
  | some f=>
    have sep:=r.final_separate checked f (by simp [hf]) tensor member
    simp only
    rw [Bytes.separate_frame _ f tensor _ sep byte bound,
      Bytes.separate_frame _ a.corrected tensor _ csep byte bound,
      Bytes.separate_frame _ a.history tensor _ hsep byte bound]

theorem exterior (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (memory : Memory Expr) (address : Nat) (history : Bytes.Outside a.history address)
    (corrected : Bytes.Outside a.corrected address)
    (final : ∀ f∈r.final.toList,Bytes.Outside f address) : publish I r a before memory address=before address := by
  unfold publish
  cases hf : r.final with
  | none=>simp only;rw [Bytes.frame _ a.corrected _ address corrected,Bytes.frame _ a.history _ address history]
  | some f=>simp only;rw [Bytes.frame _ f _ address (final f (by simp [hf])),
      Bytes.frame _ a.corrected _ address corrected,Bytes.frame _ a.history _ address history]

theorem history_word (I : Primitives) (r : Request) (checked : r.check=Bool.true)
    (a : Allocations r) (before : Bytes.Memory) (head : Nat) (hh : head<r.B*r.H)
    (chunk : Nat) (hc : chunk<r.dims.NT) (key : Nat) (hk : key<r.K) (value : Nat) (hv : value<r.V) :
    Bytes.read (returnedBytes I r a before) (a.history.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      (Native.historyGraph r.params (r.shape checked) (bits r a before) head chunk key value hk).eval I.isa := by
  have bound : ((head*r.dims.NT+chunk)*r.K+key)*r.V+value<a.history.count := by
    rw [a.history_count]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh hc) hk) hv
  rw [returnedBytes,publish_history I r a before _ _ bound]
  have val:=(source_complete r checked a before).2.history head hh chunk hc key hk value hv
  unfold words
  change source r a before "history" (((head*r.dims.NT+chunk)*r.K+key)*r.V+value)=_ at val
  rw [val,Cell.read_same]
  rfl

theorem corrected_word (I : Primitives) (r : Request) (checked : r.check=Bool.true)
    (a : Allocations r) (before : Bytes.Memory) (head : Nat) (hh : head<r.B*r.H)
    (time : Nat) (ht : time<r.T) (value : Nat) (hv : value<r.V) :
    Bytes.read (returnedBytes I r a before) (a.corrected.byteBase+4*((head*r.T+time)*r.V+value))=
      (Native.correctedGraph r.params (r.shape checked) (bits r a before) head time value).eval I.isa := by
  have bound : (head*r.T+time)*r.V+value<a.corrected.count := by
    rw [a.corrected_count]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh ht) hv
  rw [returnedBytes,publish_corrected I r a before _ _ bound]
  have val:=(source_complete r checked a before).2.corrected head hh time ht value hv
  unfold words
  dsimp only [Request.params,Request.dims,Host.Dimensions.parameters] at val
  rw [val,Cell.read_same]
  rfl

theorem final_word (I : Primitives) (r : Request) (checked : r.check=Bool.true)
    (a : Allocations r) (before : Bytes.Memory) (f : Tensor) (hf : r.final=some f)
    (head : Nat) (hh : head<r.B*r.H) (key : Nat) (hk : key<r.K) (value : Nat) (hv : value<r.V) :
    Bytes.read (returnedBytes I r a before) (f.byteBase+4*((head*r.K+key)*r.V+value))=
      (Native.historyGraph r.params (r.shape checked) (bits r a before) head r.dims.NT key value hk).eval I.isa := by
  have bound : (head*r.K+key)*r.V+value<f.count := by
    rw [r.state_count checked f (by simp [hf])]
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt
      (VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.packed_lt hh hk) hv
  rw [returnedBytes,publish_final I r a before _ f hf _ bound]
  have flag : r.params.storeFinal=Bool.true:=by simp [Request.params,Request.dims,Host.Dimensions.parameters,hf]
  have val:=(source_complete r checked a before).2.final flag head hh key hk value hv
  unfold words
  dsimp only [Request.params,Request.dims,Host.Dimensions.parameters] at val
  rw [val,Cell.read_same]
  rfl

#axiomsClean source_complete
#axiomsClean history_word
#axiomsClean corrected_word
#axiomsClean final_word
#axiomsClean caller_frame
#axiomsClean exterior
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
