import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelResult
import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelGrid
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelGridResult
open VeriTile.Triton FP.Structural KernelSource KernelExecution KernelGrid
noncomputable section
set_option maxHeartbeats 2500000

def SameInputs {α : Type} (initial current : State α) : Prop :=
  (∀ o, current.mem "input" o = initial.mem "input" o) ∧
  (∀ o, current.mem "destination" o = initial.mem "destination" o)

def advance {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) (s : State α) (pid : Nat) : State α :=
  (execution M convert equal p (atProgram s pid) hw).val

def finalState {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) : List Nat → State α → State α
  | [], s => s
  | pid :: tail, s => finalState M convert equal p hw tail (advance M convert equal p hw s pid)

theorem run_finalState {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) (order : List Nat) (s : State α) :
    KernelGrid.run (backend M convert equal) p order s =
      some (finalState M convert equal p hw order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hs := (execution M convert equal p (atProgram s pid) hw).property
    change FP.Structural.exec (backend M convert equal) (kernel p) (atProgram s pid) =
      some (advance M convert equal p hw s pid) at hs
    simpa only [KernelGrid.run, finalState, hs, Option.bind_some] using ih (advance M convert equal p hw s pid)

theorem advance_memory {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) (s : State α) (pid dst : Nat)
    (hd : (s.mem "destination" pid).read .int = (dst : Int)) :
    (advance M convert equal p hw s pid).mem =
      (KernelResult.written M convert equal p (atProgram s pid) dst).mem := by
  apply KernelResult.execution_memory
  simpa [atProgram] using hd

theorem destination_read {α : Type} [Inhabited α] (initial s : State α)
    (hs : SameInputs initial s) (pid dst : Nat)
    (hd : (initial.mem "destination" pid).read .int = (dst : Int)) :
    (s.mem "destination" pid).read .int = (dst : Int) := by rw [hs.2 pid, hd]

theorem advance_inputs {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial s : State α) (hs : SameInputs initial s) (pid dst : Nat)
    (hd : (initial.mem "destination" pid).read .int = (dst : Int)) :
    SameInputs initial (advance M convert equal p hw s pid) := by
  have hm := advance_memory M convert equal p hw s pid dst (destination_read initial s hs pid dst hd)
  constructor
  · intro o
    rw [hm, KernelResult.written_frame _ _ _ _ _ _ _ _ (Or.inl (by decide)) (Or.inl (by decide))]
    exact hs.1 o
  · intro o
    rw [hm, KernelResult.written_frame _ _ _ _ _ _ _ _ (Or.inl (by decide)) (Or.inl (by decide))]
    exact hs.2 o

theorem final_inputs {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "destination" pid).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count)
    (s : State α) (hs : SameInputs initial s) :
    SameInputs initial (finalState M convert equal p hw order s) := by
  induction order generalizing s with
  | nil => exact hs
  | cons pid tail ih =>
    exact ih (fun q hq => valid q (by simp [hq])) _
      (advance_inputs M convert equal p hw initial s hs pid (dest pid) (metadata pid (valid pid (by simp))))

theorem code_same_inputs {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (initial s : State α)
    (hs : SameInputs initial s) (pid : Nat) :
    KernelResult.code M convert equal p (atProgram s pid) =
      KernelResult.code M convert equal p (atProgram initial pid) := by
  have hl : KernelResult.loaded M p (atProgram s pid) =
      KernelResult.loaded M p (atProgram initial pid) := by
    funext i
    simp [KernelResult.loaded, atProgram, hs.1]
  funext i
  simp only [KernelResult.code, KernelResult.denominator, KernelResult.scale, KernelResult.maximum, hl]

theorem scale_same_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial s : State α) (hs : SameInputs initial s) (pid : Nat) :
    KernelResult.scale M p (atProgram s pid) = KernelResult.scale M p (atProgram initial pid) := by
  have hl : KernelResult.loaded M p (atProgram s pid) =
      KernelResult.loaded M p (atProgram initial pid) := by
    funext i
    simp [KernelResult.loaded, atProgram, hs.1]
  funext h
  simp only [KernelResult.scale, KernelResult.maximum, hl]

theorem final_frame {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "destination" pid).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count)
    (r : RegionName) (o : Nat)
    (codes : r ≠ "output" ∨ ∀ pid h c, pid < count → h < p.heads → c < p.columns →
      o ≠ KernelResult.codeOffset p (dest pid) h c)
    (scales : r ≠ "scale" ∨ ∀ pid h, pid < count → h < p.heads →
      o ≠ KernelResult.scaleOffset p (dest pid) h)
    (s : State α) (hs : SameInputs initial s) :
    (finalState M convert equal p hw order s).mem r o = s.mem r o := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    have hd := destination_read initial s hs pid (dest pid) (metadata pid hp)
    have hn := advance_inputs M convert equal p hw initial s hs pid (dest pid) (metadata pid hp)
    simp only [finalState]
    rw [ih (fun q hq => valid q (by simp [hq])) _ hn,
      advance_memory M convert equal p hw s pid (dest pid) hd]
    apply KernelResult.written_frame
    · rcases codes with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun h c hh hc => ho pid h c hp hh hc)
    · rcases scales with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun h hh => ho pid h hp hh)

theorem finalState_append {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (before after : List Nat) (s : State α) :
    finalState M convert equal p hw (before ++ after) s =
      finalState M convert equal p hw after (finalState M convert equal p hw before s) := by
  induction before generalizing s with
  | nil => rfl
  | cons pid tail ih => exact ih _

theorem final_frame_order {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "destination" pid).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (r : RegionName) (o : Nat)
    (miss : ∀ pid ∈ order,
      (r ≠ "output" ∨ ∀ h c, h < p.heads → c < p.columns → o ≠ KernelResult.codeOffset p (dest pid) h c) ∧
      (r ≠ "scale" ∨ ∀ h, h < p.heads → o ≠ KernelResult.scaleOffset p (dest pid) h))
    (s : State α) (hs : SameInputs initial s) :
    (finalState M convert equal p hw order s).mem r o = s.mem r o := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    have hd := destination_read initial s hs pid (dest pid) (metadata pid hp)
    have hn := advance_inputs M convert equal p hw initial s hs pid (dest pid) (metadata pid hp)
    simp only [finalState]
    rw [ih (fun q hq => valid q (by simp [hq]))
      (fun q hq => miss q (by simp [hq])) _ hn,
      advance_memory M convert equal p hw s pid (dest pid) hd]
    exact KernelResult.written_frame _ _ _ _ _ _ _ _ (miss pid (by simp)).1 (miss pid (by simp)).2

theorem code_slot_injective (p : Parameters) (a b h k c d : Nat)
    (hh : h < p.heads) (hk : k < p.heads) (hc : c < p.columns) (hd : d < p.columns)
    (he : KernelResult.codeOffset p a h c = KernelResult.codeOffset p b k d) : a = b := by
  have hlinear : (a * p.heads + h) * p.columns + c = (b * p.heads + k) * p.columns + d := by
    simpa [KernelResult.codeOffset, Nat.add_mul, Nat.mul_add, Nat.mul_assoc,
      Nat.mul_comm, Nat.mul_left_comm] using he
  have hr := (RowwiseQuantizationTriton.linear_injective p.columns _ _ _ _ hc hd hlinear).1
  exact (RowwiseQuantizationTriton.linear_injective p.heads _ _ _ _ hh hk hr).1

theorem scale_slot_injective (p : Parameters) (a b h k : Nat)
    (hh : h < p.heads) (hk : k < p.heads)
    (he : KernelResult.scaleOffset p a h = KernelResult.scaleOffset p b k) : a = b :=
  (RowwiseQuantizationTriton.linear_injective p.heads _ _ _ _ hh hk he).1

theorem final_code {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "destination" pid).read .int = (dest pid : Int))
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (nodup : order.Nodup)
    (q : Nat) (hq : q ∈ order) (i : TileIndex [p.height, p.width]) (hi : KernelResult.active p i) :
    (finalState M convert equal p hw order initial).mem "output"
      (KernelResult.codeOffset p (dest q) i.1.val i.2.1.val) =
      .mk .int (KernelResult.code M convert equal p (atProgram initial q) i) := by
  obtain ⟨before, after, ho⟩ := List.append_of_mem hq
  have hn := nodup
  rw [ho, List.nodup_append, List.nodup_cons] at hn
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hn
  have vb : ∀ pid ∈ before, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  have va : ∀ pid ∈ after, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  let middle := finalState M convert equal p hw before initial
  have hb : SameInputs initial middle :=
    final_inputs M convert equal p hw initial count dest metadata before vb initial ⟨fun _ => rfl, fun _ => rfl⟩
  have hd := destination_read initial middle hb q (dest q) (metadata q (valid q hq))
  have hnext := advance_inputs M convert equal p hw initial middle hb q (dest q) (metadata q (valid q hq))
  rw [ho, finalState_append, finalState]
  change (finalState M convert equal p hw after (advance M convert equal p hw middle q)).mem _ _ = _
  rw [final_frame_order M convert equal p hw initial count dest metadata after va _ _ ?_
    _ hnext, advance_memory M convert equal p hw middle q (dest q) hd,
    KernelResult.written_code M convert equal p (atProgram middle q) (dest q) i hi,
    code_same_inputs M convert equal p initial middle hb q]
  intro pid hp
  refine ⟨Or.inr ?_, Or.inl (by decide)⟩
  intro h c hh hc he
  have de := code_slot_injective p _ _ _ _ _ _ hi.1 hh hi.2 hc he
  exact hnot ((unique q pid (valid q hq) (va pid hp) de).symm ▸ hp)

theorem final_scale {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "destination" pid).read .int = (dest pid : Int))
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (nodup : order.Nodup)
    (q : Nat) (hq : q ∈ order) (i : TileIndex [p.height, 1]) (hi : i.1.val < p.heads) :
    (finalState M convert equal p hw order initial).mem "scale"
      (KernelResult.scaleOffset p (dest q) i.1.val) =
      .mk .real (KernelResult.scale M p (atProgram initial q) i.1) := by
  obtain ⟨before, after, ho⟩ := List.append_of_mem hq
  have hn := nodup
  rw [ho, List.nodup_append, List.nodup_cons] at hn
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hn
  have vb : ∀ pid ∈ before, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  have va : ∀ pid ∈ after, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  let middle := finalState M convert equal p hw before initial
  have hb : SameInputs initial middle :=
    final_inputs M convert equal p hw initial count dest metadata before vb initial ⟨fun _ => rfl, fun _ => rfl⟩
  have hd := destination_read initial middle hb q (dest q) (metadata q (valid q hq))
  have hnext := advance_inputs M convert equal p hw initial middle hb q (dest q) (metadata q (valid q hq))
  rw [ho, finalState_append, finalState]
  change (finalState M convert equal p hw after (advance M convert equal p hw middle q)).mem _ _ = _
  rw [final_frame_order M convert equal p hw initial count dest metadata after va _ _ ?_
    _ hnext, advance_memory M convert equal p hw middle q (dest q) hd,
    KernelResult.written_scale M convert equal p (atProgram middle q) (dest q) i hi,
    scale_same_inputs M p initial middle hb q]
  intro pid hp
  refine ⟨Or.inl (by decide), Or.inr ?_⟩
  intro h hh he
  have de := scale_slot_injective p _ _ _ _ hi hh he
  exact hnot ((unique q pid (valid q hq) (va pid hp) de).symm ▸ hp)

#axiomsClean run_finalState
#axiomsClean advance_inputs
#axiomsClean final_inputs
#axiomsClean code_same_inputs
#axiomsClean scale_same_inputs
#axiomsClean final_frame
#axiomsClean final_code
#axiomsClean final_scale
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelGridResult
