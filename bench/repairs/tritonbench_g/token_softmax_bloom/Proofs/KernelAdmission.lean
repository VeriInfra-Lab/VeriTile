import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelAdmission
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact checked admission, including the zero-maximum contents check. -/
theorem check_input_iff (c : Call) (s n : Nat → Int) :
    checkInput c s n=.ok () ↔ InputContract c s n := by
  have shape : InputContract c s n ↔ MetadataContract c ∧ (contentActive c → ContentChecks c s n) :=
    ⟨fun h => ⟨h.metadata,h.contents⟩,fun h => ⟨h.1,h.2⟩⟩
  rw [shape,←check_iff]
  unfold checkInput checkContents
  cases result : check c <;>
    by_cases a : contentActive c <;> by_cases b : ContentChecks c s n <;> simp_all

theorem bounds (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launched : contentActive c.observation) (b : Nat) (hb : b < rows c.start.observation) :
    0 ≤ observedStart c metadataRead b ∧ 0 ≤ observedLength c metadataRead b ∧
      length c metadataRead b ≤ c.maximum ∧ start c metadataRead b+length c metadataRead b ≤ cols c.input.observation := by
  have h := (contract.contents launched).1 b (List.mem_range.mpr hb)
  dsimp [TensorCall.observation] at h
  simp only [start,length]
  omega

theorem cover (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launched : contentActive c.observation) (b : Nat) (hb : b < rows c.start.observation) :
    length c metadataRead b ≤ (params c).width :=
  (bounds c metadataRead contract launched b hb).2.2.1.trans
    (VeriTile.Bench.Repairs.SquareMatrix.block_covers c.maximum)

theorem disjoint (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launched : contentActive c.observation) (b d : Nat)
    (hb : b < rows c.start.observation) (hd : d < rows c.start.observation)
    (different : b≠d) (bn : 0 < length c metadataRead b) (dn : 0 < length c metadataRead d) :
    start c metadataRead b+length c metadataRead b ≤ start c metadataRead d ∨
      start c metadataRead d+length c metadataRead d ≤ start c metadataRead b := by
  have bb := bounds c metadataRead contract launched b hb
  have dd := bounds c metadataRead contract launched d hd
  have apart := (contract.contents launched).2 b (List.mem_range.mpr hb) d (List.mem_range.mpr hd) different
  unfold length start at *
  have result := apart (by omega) (by omega)
  omega

theorem separate (c : TensorCall) : Grid.Separate (params c) := by
  intro r hr
  change r≠("output" : RegionName)
  simp only [Grid.Reads,params,Region.cast] at hr
  rcases hr with rfl|rfl|rfl <;> decide

theorem output_injective (c : TensorCall)
    (contract : MetadataContract c.observation)
    (launched : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (h t k u : Nat) (hh : h < rows c.input.observation) (ht : t < cols c.input.observation)
    (hk : k < rows c.input.observation) (hu : u < cols c.input.observation)
    (same : h*c.output.rowStride+t*c.output.colStride=k*c.output.rowStride+u*c.output.colStride) :
    h=k ∧ t=u :=
  nonoverlap_injective _ _ _ _ _ _ _ _ (contract.launch launched).2.2.2.1 hh ht hk hu same

theorem unique {α} [Inhabited α] (c : TensorCall) (read : Nat → α) (metadataRead : Nat → Word)
    (before : NativeCells α)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launched : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (work : List Grid.PID)
    (complete : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    Grid.Unique (params c) (view c read metadataRead before) work := by
  intro e he f hf same
  obtain ⟨em,ea⟩ := (Grid.mem_events _ _ _ _).mp he
  obtain ⟨fm,fa⟩ := (Grid.mem_events _ _ _ _).mp hf
  have eh := (Grid.mem_fullGrid _ _ _).mp (complete.mem_iff.mpr em)
  have fh := (Grid.mem_fullGrid _ _ _).mp (complete.mem_iff.mpr fm)
  have content : contentActive c.observation := ⟨launched.1,launched.2.1,launched.2.2.1⟩
  have ei : e.2.1.val < length c metadataRead e.1.1 := by simpa [Grid.active] using ea
  have fi : f.2.1.val < length c metadataRead f.1.1 := by simpa [Grid.active] using fa
  have eb : start c metadataRead e.1.1+e.2.1.val < cols c.input.observation :=
    lt_of_lt_of_le (Nat.add_lt_add_left ei _) (bounds c metadataRead contract content e.1.1 eh.1).2.2.2
  have fb : start c metadataRead f.1.1+f.2.1.val < cols c.input.observation :=
    lt_of_lt_of_le (Nat.add_lt_add_left fi _) (bounds c metadataRead contract content f.1.1 fh.1).2.2.2
  change e.1.2*c.output.rowStride+(start c metadataRead e.1.1+e.2.1.val)*c.output.colStride=
    f.1.2*c.output.rowStride+(start c metadataRead f.1.1+f.2.1.val)*c.output.colStride at same
  have inner := output_injective c contract.metadata launched _ _ _ _ eh.2 eb fh.2 fb same
  have batch : e.1.1=f.1.1 := by
    by_contra different
    have apart := disjoint c metadataRead contract content _ _ eh.1 fh.1 different (by omega) (by omega)
    rcases apart with le|le <;> omega
  apply Prod.ext
  · exact Prod.ext batch inner.1
  · apply Prod.ext
    · apply Fin.ext
      have equal := inner.2
      rw [batch] at equal
      omega
    · exact Subsingleton.elim _ _

#axiomsClean check_input_iff
#axiomsClean unique
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelAdmission
