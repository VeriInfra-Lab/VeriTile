import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.NativeMemory
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

/-- Ordinary saved-Tensor version facts compare scalar reads of two actual
memory states. At checkpoint level one no saved-history equality is assumed;
the source recomputes that history from the saved primal tensors and seed. -/
structure Versions (c : Shape) (finished later : Heap α) (checkpoint : Bool) : Prop where
  q : (inputsOfHeap later).q=(inputsOfHeap finished).q
  k : (inputsOfHeap later).k=(inputsOfHeap finished).k
  v : (inputsOfHeap later).v=(inputsOfHeap finished).v
  seed : (inputsOfHeap later).seed=(inputsOfHeap finished).seed
  history : checkpoint=Bool.false → ∀ h<c.heads,∀ t<c.chunks,∀ e<c.K,∀ j<c.V,
    later ⟨.history,h,t,e,j⟩=finished ⟨.history,h,t,e,j⟩

def recomputeCode (c : Shape) (tree : SumTree) (checkpoint : Bool) : List Store :=
  if checkpoint then phaseCode c tree .h (Source.eventsH c Bool.false) else []
def prepareBackward (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (checkpoint : Bool) (later : Heap α) : Heap α :=
  if checkpoint then Source.afterH ops c scale tree Bool.false later else later

theorem prepare_inputs (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (checkpoint : Bool) (later : Heap α) :
    inputsOfHeap (prepareBackward ops c scale tree checkpoint later)=inputsOfHeap later := by
  cases checkpoint
  · rfl
  · exact Source.h_inputs ops c scale tree Bool.false later

theorem history_inputs (ops : Ops α) (c : Shape) (a b : Inputs α)
    (hk : a.k=b.k) (hv : a.v=b.v) (hs : a.seed=b.seed) (h n : Nat) :
    Reference.history ops c a h n=Reference.history ops c b h n := by
  induction n with
  | zero => simp [Reference.history,hs]
  | succ n ih =>
    simp only [Reference.history,ih]
    funext e j
    simp only [Body.nextH,kAt,vAt,hk,hv]

theorem history_preserved (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf : Bool) (before : Heap α) (h t e j : Nat) :
    Source.afterO ops c scale tree rf before ⟨.history,h,t,e,j⟩=
      Source.afterH ops c scale tree rf before ⟨.history,h,t,e,j⟩ := by
  exact Source.execute_frame ops c scale tree .o _ (Source.o_footprint c) _ _ (by simp [Phase.writes])

theorem saved_from_versions (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf checkpoint : Bool) (before finished later : Heap α)
    (completed : finished=Source.afterO ops c scale tree rf before)
    (versions : Versions c finished later checkpoint) :
    Separate.Saved ops c scale tree rf before (prepareBackward ops c scale tree checkpoint later) := by
  have inputs := Source.o_inputs ops c scale tree rf before
  have hq : (inputsOfHeap later).q=(inputsOfHeap before).q := by rw [versions.q,completed,inputs]
  have hk : (inputsOfHeap later).k=(inputsOfHeap before).k := by rw [versions.k,completed,inputs]
  have hv : (inputsOfHeap later).v=(inputsOfHeap before).v := by rw [versions.v,completed,inputs]
  have hs : (inputsOfHeap later).seed=(inputsOfHeap before).seed := by rw [versions.seed,completed,inputs]
  constructor
  · rwa [prepare_inputs]
  · rwa [prepare_inputs]
  · rwa [prepare_inputs]
  · intro h hh t ht e he j hj
    cases checkpoint
    · exact (versions.history rfl h hh t ht e he j hj).trans (congrFun completed _)
    · change Source.afterH ops c scale tree Bool.false later _=Source.afterO ops c scale tree rf before _
      rw [history_preserved]
      rw [Source.h_value ops c scale tree Bool.false later h t e j hh ht he hj,
        Source.h_value ops c scale tree rf before h t e j hh ht he hj,
        history_inputs ops c _ _ hk hv hs]

theorem prepare_physical (ops : Ops α) (c : Shape) (storage : Storage c) (scale : α)
    (tree : SumTree) (checkpoint : Bool) (later : Memory α) :
    storage.binding.view ops.zero (execute ops c storage.binding scale (recomputeCode c tree checkpoint) later)=
      prepareBackward ops c scale tree checkpoint (storage.binding.view ops.zero later) := by
  cases checkpoint
  · rfl
  · simp only [recomputeCode,↓reduceIte,execute_phase,prepareBackward,Source.afterH]
    exact Source.execute_physical ops c storage.binding scale tree .h _ (Source.h_valid c Bool.false)
      (fun a ha=>Source.write_not_input .h a (Source.h_footprint c Bool.false a ha)) later

theorem recompute_stores_valid (c : Shape) (tree : SumTree) (checkpoint : Bool) (s : Store)
    (hs : s∈recomputeCode c tree checkpoint) : s.address.Valid c ∧ s.address.region∉inputRegions := by
  cases checkpoint
  · simp [recomputeCode] at hs
  · have mem : s.address∈Source.eventsH c Bool.false := by
      rw [←phaseCode_addresses c tree .h (Source.eventsH c Bool.false)]
      exact List.mem_map.mpr ⟨s,hs,rfl⟩
    exact ⟨Source.h_valid c Bool.false _ mem,
      Source.write_not_input .h _ (Source.h_footprint c Bool.false _ mem)⟩

theorem backward_saved (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (rf ri checkpoint : Bool)
    (forwardStorage backwardStorage : Storage c) (before finished later : Memory α)
    (completed : forwardStorage.binding.view ops.zero finished=
      Source.afterO ops c scale tree rf (forwardStorage.binding.view ops.zero before))
    (versions : Versions c (forwardStorage.binding.view ops.zero finished)
      (backwardStorage.binding.view ops.zero later) checkpoint) :
    Separate.BackwardPost ops c
      (Separate.joined (forwardStorage.binding.view ops.zero before) (backwardStorage.binding.view ops.zero later))
      scale tree (backwardStorage.binding.view ops.zero
        (execute ops c backwardStorage.binding scale
          (recomputeCode c tree checkpoint++backwardCode c tree ri) later)) := by
  rw [execute_append,backward_graph,Separate.backward_physical,prepare_physical]
  have saved := saved_from_versions ops c scale tree rf checkpoint _ _ _ completed versions
  have post := Separate.backwardAPI_post ops c scale tree complete rf ri _ _ saved
  have joined : Separate.joined (forwardStorage.binding.view ops.zero before)
      (prepareBackward ops c scale tree checkpoint (backwardStorage.binding.view ops.zero later))=
      Separate.joined (forwardStorage.binding.view ops.zero before) (backwardStorage.binding.view ops.zero later) := by
    unfold Separate.joined
    rw [prepare_inputs]
  rwa [joined] at post

#axiomsClean saved_from_versions
#axiomsClean backward_saved
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
