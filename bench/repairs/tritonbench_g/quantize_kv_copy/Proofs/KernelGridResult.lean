import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelResult
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelGrid
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelGridResult
open VeriTile.Triton FP.Structural Kernel KernelGrid
noncomputable section
set_option maxHeartbeats 2500000

def target (p : Parameters) (dest : Nat → Nat) (pid : Nat) : Nat := dest pid*p.heads+pid%p.heads

def SameInputs {α : Type} (initial current : State α) : Prop :=
  (∀ o, current.mem "K" o = initial.mem "K" o) ∧
  (∀ o, current.mem "D" o = initial.mem "D" o)

def advance {α : Type} [Inhabited α] (op : Ops α) (p : Parameters)  (s : State α) (pid : Nat) : State α :=
  (execution op p (atProgram p s pid) ).val

def finalState {α : Type} [Inhabited α] (op : Ops α) (p : Parameters)  : List Nat → State α → State α
  | [], s => s
  | pid :: tail, s => finalState op p tail (advance op p s pid)

theorem run_finalState {α : Type} [Inhabited α] (op : Ops α) (p : Parameters)  (order : List Nat) (s : State α) :
    KernelGrid.run op p order s =
      some (finalState op p order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hs := (execution op p (atProgram p s pid) ).property
    change FP.Structural.exec (backend op) (kernel p) (atProgram p s pid) =
      some (advance op p s pid) at hs
    simpa only [KernelGrid.run, finalState, hs, Option.bind_some] using ih (advance op p s pid)

theorem advance_memory {α : Type} [Inhabited α] (op : Ops α) (p : Parameters)  (s : State α) (pid dst : Nat)
    (hd : (s.mem "D" (pid/p.heads)).read .int = (dst : Int)) :
    (advance op p s pid).mem =
      (Kernel.written op p (atProgram p s pid) (dst*p.heads+pid%p.heads)).mem := by
  simpa [advance,atProgram] using Kernel.execution_memory op p (atProgram p s pid) dst (by simpa [atProgram] using hd)

theorem destination_read {α : Type} [Inhabited α] (p : Parameters) (initial s : State α)
    (hs : SameInputs initial s) (pid dst : Nat)
    (hd : (initial.mem "D" (pid/p.heads)).read .int = (dst : Int)) :
    (s.mem "D" (pid/p.heads)).read .int = (dst : Int) := by rw [hs.2 (pid/p.heads), hd]

theorem advance_inputs {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial s : State α) (hs : SameInputs initial s) (pid dst : Nat)
    (hd : (initial.mem "D" (pid/p.heads)).read .int = (dst : Int)) :
    SameInputs initial (advance op p s pid) := by
  have hm := advance_memory op p s pid dst (destination_read p initial s hs pid dst hd)
  constructor
  · intro o
    rw [hm, Kernel.written_frame _ _ _ _ _ _ (Or.inl (by decide)) (Or.inl (by decide))]
    exact hs.1 o
  · intro o
    rw [hm, Kernel.written_frame _ _ _ _ _ _ (Or.inl (by decide)) (Or.inl (by decide))]
    exact hs.2 o

theorem final_inputs {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "D" (pid/p.heads)).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count)
    (s : State α) (hs : SameInputs initial s) :
    SameInputs initial (finalState op p order s) := by
  induction order generalizing s with
  | nil => exact hs
  | cons pid tail ih =>
    exact ih (fun q hq => valid q (by simp [hq])) _
      (advance_inputs op p initial s hs pid (dest pid) (metadata pid (valid pid (by simp))))

theorem code_same_inputs {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) (initial s : State α)
    (hs : SameInputs initial s) (pid : Nat) :
    Kernel.code op p (atProgram p s pid) =
      Kernel.code op p (atProgram p initial pid) := by
  have hl : Kernel.loaded op p (atProgram p s pid) =
      Kernel.loaded op p (atProgram p initial pid) := by
    funext i
    simp [Kernel.loaded, atProgram, hs.1]
  funext i
  simp only [Kernel.code, Kernel.divisor, Kernel.scale, Kernel.maximum, hl]

theorem scale_same_inputs {α : Type} [Inhabited α] (op : Ops α)
    (p : Parameters) (initial s : State α) (hs : SameInputs initial s) (pid : Nat) :
    Kernel.scale op p (atProgram p s pid) = Kernel.scale op p (atProgram p initial pid) := by
  have hl : Kernel.loaded op p (atProgram p s pid) =
      Kernel.loaded op p (atProgram p initial pid) := by
    funext i
    simp [Kernel.loaded, atProgram, hs.1]
  funext h
  simp only [Kernel.scale, Kernel.maximum, hl]

theorem final_frame {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "D" (pid/p.heads)).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count)
    (r : RegionName) (o : Nat)
    (codes : r ≠ "O" ∨ ∀ pid h c, pid < count → h < p.groups → c < 8 →
      o ≠ Kernel.codeOffset p (target p dest pid) h c)
    (scales : r ≠ "S" ∨ ∀ pid h, pid < count → h < p.groups →
      o ≠ Kernel.scaleOffset p (target p dest pid) h)
    (s : State α) (hs : SameInputs initial s) :
    (finalState op p order s).mem r o = s.mem r o := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    have hd := destination_read p initial s hs pid (dest pid) (metadata pid hp)
    have hn := advance_inputs op p initial s hs pid (dest pid) (metadata pid hp)
    simp only [finalState]
    rw [ih (fun q hq => valid q (by simp [hq])) _ hn,
      advance_memory op p s pid (dest pid) hd]
    apply Kernel.written_frame
    · rcases codes with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun h c hh hc => ho pid h c hp hh hc)
    · rcases scales with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun h hh => ho pid h hp hh)

theorem finalState_append {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (before after : List Nat) (s : State α) :
    finalState op p (before ++ after) s =
      finalState op p after (finalState op p before s) := by
  induction before generalizing s with
  | nil => rfl
  | cons pid tail ih => exact ih _

theorem final_frame_order {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "D" (pid/p.heads)).read .int = (dest pid : Int))
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (r : RegionName) (o : Nat)
    (miss : ∀ pid ∈ order,
      (r ≠ "O" ∨ ∀ h c, h < p.groups → c < 8 → o ≠ Kernel.codeOffset p (target p dest pid) h c) ∧
      (r ≠ "S" ∨ ∀ h, h < p.groups → o ≠ Kernel.scaleOffset p (target p dest pid) h))
    (s : State α) (hs : SameInputs initial s) :
    (finalState op p order s).mem r o = s.mem r o := by
  induction order generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    have hp := valid pid (by simp)
    have hd := destination_read p initial s hs pid (dest pid) (metadata pid hp)
    have hn := advance_inputs op p initial s hs pid (dest pid) (metadata pid hp)
    simp only [finalState]
    rw [ih (fun q hq => valid q (by simp [hq]))
      (fun q hq => miss q (by simp [hq])) _ hn,
      advance_memory op p s pid (dest pid) hd]
    exact Kernel.written_frame _ _ _ _ _ _ (miss pid (by simp)).1 (miss pid (by simp)).2

theorem code_slot_injective (p : Parameters) (a b h k c d : Nat)
    (hh : h < p.groups) (hk : k < p.groups) (hc : c < 8) (hd : d < 8)
    (he : Kernel.codeOffset p a h c = Kernel.codeOffset p b k d) : a = b := by
  have hlinear : (a * p.groups + h) * 8 + c = (b * p.groups + k) * 8 + d := by
    simpa [Kernel.codeOffset, Nat.add_mul, Nat.mul_add, Nat.mul_assoc,
      Nat.mul_comm, Nat.mul_left_comm] using he
  have hr := (RowwiseQuantizationTriton.linear_injective 8 _ _ _ _ hc hd hlinear).1
  exact (RowwiseQuantizationTriton.linear_injective p.groups _ _ _ _ hh hk hr).1

theorem scale_slot_injective (p : Parameters) (a b h k : Nat)
    (hh : h < p.groups) (hk : k < p.groups)
    (he : Kernel.scaleOffset p a h = Kernel.scaleOffset p b k) : a = b :=
  (RowwiseQuantizationTriton.linear_injective p.groups _ _ _ _ hh hk he).1

theorem final_code {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "D" (pid/p.heads)).read .int = (dest pid : Int))
    (unique : ∀ a b, a < count → b < count → target p dest a = target p dest b → a = b)
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (nodup : order.Nodup)
    (q : Nat) (hq : q ∈ order) (i : TileIndex [p.block, 8]) (hi : i.1.val < p.groups) :
    (finalState op p order initial).mem "O"
      (Kernel.codeOffset p (target p dest q) i.1.val i.2.1.val) =
      .mk .int (Kernel.code op p (atProgram p initial q) i) := by
  obtain ⟨before, after, ho⟩ := List.append_of_mem hq
  have hn := nodup
  rw [ho, List.nodup_append, List.nodup_cons] at hn
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hn
  have vb : ∀ pid ∈ before, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  have va : ∀ pid ∈ after, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  let middle := finalState op p before initial
  have hb : SameInputs initial middle :=
    final_inputs op p initial count dest metadata before vb initial ⟨fun _ => rfl, fun _ => rfl⟩
  have hd := destination_read p initial middle hb q (dest q) (metadata q (valid q hq))
  have hnext := advance_inputs op p initial middle hb q (dest q) (metadata q (valid q hq))
  rw [ho, finalState_append, finalState]
  change (finalState op p after (advance op p middle q)).mem _ _ = _
  rw [final_frame_order op p initial count dest metadata after va _ _ ?_
    _ hnext, advance_memory op p middle q (dest q) hd]
  change (Kernel.written op p (atProgram p middle q) (target p dest q)).mem _ _ = _
  rw [Kernel.written_code op p (atProgram p middle q) (target p dest q) i hi,
    code_same_inputs op p initial middle hb q]
  intro pid hp
  refine ⟨Or.inr ?_, Or.inl (by decide)⟩
  intro h c hh hc he
  have de := code_slot_injective p _ _ _ _ _ _ hi hh i.2.1.isLt hc he
  exact hnot ((unique q pid (valid q hq) (va pid hp) de).symm ▸ hp)

theorem final_scale {α : Type} [Inhabited α] (op : Ops α) (p : Parameters) 
    (initial : State α) (count : Nat) (dest : Nat → Nat)
    (metadata : ∀ pid, pid < count → (initial.mem "D" (pid/p.heads)).read .int = (dest pid : Int))
    (unique : ∀ a b, a < count → b < count → target p dest a = target p dest b → a = b)
    (order : List Nat) (valid : ∀ pid ∈ order, pid < count) (nodup : order.Nodup)
    (q : Nat) (hq : q ∈ order) (i : TileIndex [p.block]) (hi : i.1.val < p.groups) :
    (finalState op p order initial).mem "S"
      (Kernel.scaleOffset p (target p dest q) i.1.val) =
      .mk .real (Kernel.scale op p (atProgram p initial q) i.1) := by
  obtain ⟨before, after, ho⟩ := List.append_of_mem hq
  have hn := nodup
  rw [ho, List.nodup_append, List.nodup_cons] at hn
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hn
  have vb : ∀ pid ∈ before, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  have va : ∀ pid ∈ after, pid < count := by
    intro pid hp; apply valid pid; rw [ho]; simp [hp]
  let middle := finalState op p before initial
  have hb : SameInputs initial middle :=
    final_inputs op p initial count dest metadata before vb initial ⟨fun _ => rfl, fun _ => rfl⟩
  have hd := destination_read p initial middle hb q (dest q) (metadata q (valid q hq))
  have hnext := advance_inputs op p initial middle hb q (dest q) (metadata q (valid q hq))
  rw [ho, finalState_append, finalState]
  change (finalState op p after (advance op p middle q)).mem _ _ = _
  rw [final_frame_order op p initial count dest metadata after va _ _ ?_
    _ hnext, advance_memory op p middle q (dest q) hd]
  change (Kernel.written op p (atProgram p middle q) (target p dest q)).mem _ _ = _
  rw [Kernel.written_scale op p (atProgram p middle q) (target p dest q) i hi,
    scale_same_inputs op p initial middle hb q]
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
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelGridResult
