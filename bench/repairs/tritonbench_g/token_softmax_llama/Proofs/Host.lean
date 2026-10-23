import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.HostExecution
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Bounds
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Native
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

/-- Full successful structural host execution, complete active output trace,
unchanged caller output object, None return and physical active-address bounds. -/
theorem host_typed {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (work : List PID) (m : Memory α) (layout : Layout c) (data : Data c m)
    (storage : Storage c) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, invoke M c work m=some result ∧ result.returned=none ∧ result.output=c.output ∧
      result.memory=(events (params c) m work).foldl (Grid.write (params c) m (values M (params c) m)) m ∧
      (∀ e∈outputEvents c m,
        result.memory c.output.region (offset (params c) m e.1 e.2)=.mk .fp32 (values M (params c) m e)) ∧
      Frame c m result.memory ∧ Accesses c m := by
  obtain ⟨result,run,ret,out,trace,readback,frame⟩ := whole_typed M c work m layout data legal
  exact ⟨result,run,ret,out,trace,readback,frame,accesses c m layout data storage⟩

theorem host_real (c : Call) (work : List PID) (m : Memory Real.Scalar)
    (layout : Layout c) (data : Data c m) (numeric : NumericData c m) (storage : Storage c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, invoke Real.algebra c work m=some result ∧ result.returned=none ∧ result.output=c.output ∧
      (∀ b < batches c, ∀ h < heads c, ∀ n < seq c m b,
        WithBot.unbotD 0 ((result.memory c.output.region
          (h*c.output.rowStride+(startAt c m b+n)*c.output.colStride)).read .fp32)=target c m b h n) ∧
      Frame c m result.memory ∧ Accesses c m := by
  obtain ⟨result,run,ret,out,_,readback,frame,safe⟩ := host_typed Real.algebra c work m layout data storage legal
  refine ⟨result,run,ret,out,?_,frame,safe⟩
  intro b hb h hh n hn
  have cover : seq c m b ≤ width c := (data.sufficientMaximum b hb).trans (width_cover c)
  have nw : n < width c := lt_of_lt_of_le hn cover
  let i : Fin (params c).width := ⟨n,nw⟩
  let e : Event (params c) := ((b,h),(i,PUnit.unit))
  have live : e∈outputEvents c m := (events_exact c m e).mpr ⟨⟨hb,hh⟩,hn⟩
  have cell := readback e live
  rw [offset_strided c m] at cell
  rw [cell,Cell.read_same]
  change WithBot.unbotD 0 (outputValues Real.algebra .fp32 (params c) (state m (b,h)) (i,PUnit.unit))=_
  rw [Real.output_target _ _ (lt_of_le_of_lt (Nat.zero_le n) nw)
    (numeric_row c m numeric b h hb hh) (by simpa using cover) i (by simpa using hn),target_eq]

@[simp] theorem seq_entry (c : Call) (m : Memory Native.Word) (b : Nat) :
    seq c (Native.entry m) b=seq c m b := by simp [seq]
@[simp] theorem start_entry (c : Call) (m : Memory Native.Word) (b : Nat) :
    startAt c (Native.entry m) b=startAt c m b := by simp [startAt]

theorem entry_cell (m : Memory Native.Word) (r : RegionName) (o : Nat)
    (d : TileDType) (v : Value Native.Word d) (h : m r o=.mk d v) :
    Native.entry m r o=.mk d (Native.inject d v) := by
  unfold Native.entry
  rw [h]

theorem data_entry (c : Call) (m : Memory Native.Word) (data : Data c m) : Data c (Native.entry m) := by
  constructor
  · simpa using data.sufficientMaximum
  · simpa using data.window
  · simpa using data.disjoint
  · simpa using data.startInt32
  · simpa using data.lengthInt32
  · intro b hb
    obtain ⟨hs,hl⟩ := data.metadataCells b hb
    simp only [start_entry,seq_entry]
    exact ⟨entry_cell m _ _ .nat _ hs,entry_cell m _ _ .nat _ hl⟩
  · intro h hh t ht
    obtain ⟨x,hx⟩ := data.inputCells h hh t ht
    exact ⟨SoftmaxTriton1.NativeGraph.atom x,entry_cell m _ _ .fp32 x hx⟩

/-- The complete ordered native graph is proved for all active host results.
Only the six local scalar instruction interpretations remain a numerical boundary. -/
theorem host_native (I : Native.Instructions) (V : Nat)
    (c : Call) (work : List PID) (m : Memory Native.Word) (layout : Layout c) (data : Data c m)
    (storage : Storage c) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, invoke (SoftmaxTriton1.NativeGraph.algebra I V) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ b < batches c, ∀ h < heads c, ∀ n < seq c m b,
        ((result.memory c.output.region (h*c.output.rowStride+(startAt c m b+n)*c.output.colStride)).read .fp32)
          (SoftmaxTriton1.NativeGraph.owner V (width c) n)=Native.target I V (params c) m (b,h) n) ∧
      Frame c (Native.entry m) result.memory ∧ Accesses c (Native.entry m) := by
  obtain ⟨result,run,ret,out,_,readback,frame,safe⟩ := host_typed (SoftmaxTriton1.NativeGraph.algebra I V)
    c work (Native.entry m) layout (data_entry c m data) storage legal
  refine ⟨result,run,ret,out,?_,frame,safe⟩
  intro b hb h hh n hn
  have nw : n < width c := lt_of_lt_of_le hn ((data.sufficientMaximum b hb).trans (width_cover c))
  let i : Fin (params c).width := ⟨n,nw⟩
  let e : Event (params c) := ((b,h),(i,PUnit.unit))
  have live : e∈outputEvents c (Native.entry m) := (events_exact c (Native.entry m) e).mpr
    ⟨⟨hb,hh⟩,by simpa [e,i] using hn⟩
  have cell := readback e live
  rw [offset_strided c (Native.entry m)] at cell
  simp only [start_entry] at cell
  rw [cell,Cell.read_same]
  exact Native.value_graph I V (params c) m (b,h) i

/-- Coverage and frame require no concrete scalar interpretation. -/
theorem host_native_coverage {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (work : List PID) (m : Memory α) (layout : Layout c) (data : Data c m)
    (storage : Storage c) (legal : (fullGrid (batches c) (heads c)).Perm work) :
    ∃ result, invoke M c work m=some result ∧ result.returned=none ∧ result.output=c.output ∧
      (∀ b < batches c, ∀ h < heads c, ∀ n < seq c m b,
        ∃ value, result.memory c.output.region (h*c.output.rowStride+(startAt c m b+n)*c.output.colStride)=.mk .fp32 value) ∧
      Frame c m result.memory ∧ Accesses c m := by
  obtain ⟨result,run,ret,out,_,readback,frame,safe⟩ := host_typed M c work m layout data storage legal
  refine ⟨result,run,ret,out,?_,frame,safe⟩
  intro b hb h hh n hn
  have nw : n < width c := lt_of_lt_of_le hn ((data.sufficientMaximum b hb).trans (width_cover c))
  let e : Event (params c) := ((b,h),(⟨n,nw⟩,PUnit.unit))
  have cell := readback e ((events_exact c m e).mpr ⟨⟨hb,hh⟩,hn⟩)
  rw [offset_strided c m] at cell
  exact ⟨values M (params c) m e,cell⟩

#axiomsClean host_real
#axiomsClean host_typed
#axiomsClean host_native
#axiomsClean host_native_coverage
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Host
