import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Allocation

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 4000000

def OutputPost (s : Shape) (provided : Option Descriptor) (caller : Nat → Prop)
    (before : Memory (Cell α)) (result : HostReturn α) (value : Fin s.M → Fin s.N → α) : Prop :=
  result.tensor.rows=s.M ∧ result.tensor.columns=s.N ∧
  (∀ out,provided=some out → result.tensor=out) ∧
  (∀ r n,result.memory (result.tensor.location.addr (r.val*s.N+n.val))=.fp32 (value r n)) ∧
  (∀ p,caller p.1 → ¬(result.tensor.location.window s.M s.N).inside p → result.memory p=before p) ∧
  (provided=none → ∀ p,caller p.1 → result.memory p=before p)

theorem host_word_post (a : Arithmetic α) (tile : Tile) (plan : LibraryPlan s)
    (allocation : Allocated s provided fresh b before after) (x : Input α s)
    (rep : Represents x b before) :
    OutputPost s provided allocation.caller before
      (hostRun a s b tile plan provided fresh after) (wordReference a x plan) := by
  refine ⟨allocation.selectedRows,allocation.selectedColumns,?_,?_,?_,?_⟩
  · intro out same
    subst provided
    rfl
  · intro r n
    simp only [hostRun,allocation.selectedLocation]
    rw [output a s b tile plan allocation.separate after r.val n.val r.isLt n.isLt]
    rw [result_words a x b plan after (allocation_represents allocation x rep) r n]
  · intro p caller outside
    apply caller_frame a tile plan allocation p caller
    simpa only [hostRun,allocation.selectedLocation] using outside
  · intro empty p caller
    subst provided
    exact fresh_caller_frame a tile plan allocation p caller

#axiomsClean host_word_post
end
end VeriTile.Bench.Repairs.DequantizeMatmul
