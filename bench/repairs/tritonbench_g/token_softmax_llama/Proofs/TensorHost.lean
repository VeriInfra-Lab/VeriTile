import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Addresses

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Triton FP.Structural Strided.Grid
noncomputable section

def exterior {α : Type} [Inhabited α] (c : TensorCall) (m : Memory α) (r : RegionName) (i : Nat) :=
  r≠storageRegion c.output.tensor.storage ∨
    ∀ e∈Strided.Host.outputEvents (host c.observation) (view c m),
      i≠c.output.tensor.storageOffset+offset (Strided.Host.params (host c.observation)) (view c m) e.1 e.2

theorem coordinate_active {α : Type} [Inhabited α] (c : TensorCall) (m : Memory α)
    (data : Strided.Host.Data (host c.observation) (view c m))
    (b h n : Nat) (hb : b  <  rows c.start.observation) (hh : h  <  rows c.input.observation)
    (hn : n  <  Strided.Host.seq (host c.observation) (view c m) b) : active c.observation := by
  have a := data.sufficientMaximum b hb
  have w := data.window b hb
  change _  ≤  c.maximum at a
  change _  ≤  cols c.input.observation at w
  exact ⟨by change 0  <  rows c.start.observation;omega,
    by change 0  <  rows c.input.observation;omega,
    by change 0  <  cols c.input.observation;omega,
    by change 0  <  c.maximum;omega⟩

def outputIndex {α : Type} [Inhabited α] (c : TensorCall) (m : Memory α) (b h n : Nat) :=
  c.output.tensor.storageOffset+
    (h*c.output.rowStride+(Strided.Host.startAt (host c.observation) (view c m) b+n)*c.output.colStride)

theorem view_map {α β : Type} (f : Cell α → Cell β) (c : TensorCall) (m : Memory α) :
    view c (fun r i => f (m r i))=fun r i => f (view c m r i) := by
  funext r i
  simp only [view]
  split_ifs <;> rfl

theorem view_entry (c : TensorCall) (m : Memory Strided.Native.Word) :
    view c (Strided.Native.entry m)=Strided.Native.entry (view c m) :=
  view_map (fun cell => match cell with | .mk d v => .mk d (Strided.Native.inject d v)) c m


/-- Every actual formed memory access is bounded by the API-certified storage.
The checked launch contract derives the layout and capacity premises. -/
theorem checked_accesses {α : Type} [Inhabited α] (c : TensorCall) (m : Memory α)
    (h : MetadataContract c.observation) (launch : active c.observation)
    (data : Strided.Host.Data (host c.observation) (view c m)) :
    Strided.Host.Accesses (host c.observation) (view c m) :=
  Strided.Host.accesses _ _ (checked_layout _ h launch) data (checked_storage c h launch)

#axiomsClean checked_accesses
#axiomsClean coordinate_active
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
