import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked_Replay

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000

def Event.Safe (allowed : Address → Prop) : Event α β → Prop
  | .storeHalf p _ => allowed p
  | _ => True

def Code.Safe (allowed : Address → Prop) : Code α β → Prop
  | .pure _ => True
  | .event e next => Event.Safe allowed e ∧ ∀ x, Code.Safe allowed (next x)

theorem safe_bind (code : Code α β) (next : β → Code α γ)
    (first : Code.Safe allowed code) (rest : ∀ x, Code.Safe allowed (next x)) :
    Code.Safe allowed (code.bind next) := by
  induction code with
  | pure x => exact rest x
  | event e next ih => exact ⟨first.1,fun x => ih x _ (first.2 x) rest⟩

theorem safe_emit (e : Event α β) (h : Event.Safe allowed e) : Code.Safe allowed (emit e) :=
  ⟨h,fun _ => trivial⟩

theorem safe_gather (f : Fin n → Code α β) (h : ∀ i, Code.Safe allowed (f i)) : Code.Safe allowed (gather f) := by
  induction n with
  | zero => trivial
  | succ n ih => exact safe_bind _ _ (h 0) (fun _ =>
      safe_bind _ _ (ih (fun i => f i.succ) (fun i => h i.succ)) (fun _ => trivial))

theorem safe_fold (f : β → α → Code α α) (h : ∀ i z, Code.Safe allowed (f i z)) (items : List β) (z : α) :
    Code.Safe allowed (foldCode f items z) := by
  induction items generalizing z with
  | nil => trivial
  | cons i rest ih => exact safe_bind _ _ (h i z) ih

theorem safe_loadA (p : Problem) (b : Buffers) (tile iteration : Nat) (row : Fin 128) (k : Fin 64) :
    Code.Safe allowed (loadACode (α:=α) p b tile iteration row k) := by
  unfold loadACode
  split <;> exact safe_emit _ trivial

theorem safe_loadB (p : Problem) (b : Buffers) (tile iteration : Nat) (k : Fin 64) (column : Fin 256) :
    Code.Safe allowed (loadBCode (α:=α) p b tile iteration k column) := by
  unfold loadBCode
  split <;> exact safe_emit _ trivial

theorem safe_block (p : Problem) (b : Buffers) (tile iteration : Nat)
    (row : Fin 128) (column : Fin 256) (z : α) : Code.Safe allowed (blockCode p b tile iteration row column z) := by
  apply safe_fold
  intro group seed
  exact safe_bind _ _ (safe_gather _ (fun _ => safe_loadA _ _ _ _ _ _)) (fun _ =>
    safe_bind _ _ (safe_gather _ (fun _ => safe_loadB _ _ _ _ _ _)) (fun _ => safe_emit _ trivial))

theorem safe_accumulator (p : Problem) (b : Buffers) (tile count : Nat)
    (row : Fin 128) (column : Fin 256) : Code.Safe allowed (accumulatorCode (α:=α) p b tile row column count) := by
  induction count with
  | zero => exact safe_emit _ trivial
  | succ count ih => exact safe_bind _ _ ih (fun _ => safe_block _ _ _ _ _ _ _)

theorem safe_activation (enabled : Bool) (value : α) : Code.Safe allowed (activationCode enabled value) := by
  cases enabled <;> simp [activationCode,bind,Code.bind,emit,Code.Safe,Event.Safe,pure]

theorem safe_value (p : Problem) (b : Buffers) (tile : Nat) (point : Address) :
    Code.Safe allowed (valueCode (α:=α) p b tile point) := by
  unfold valueCode
  split
  · exact safe_bind _ _ (safe_accumulator _ _ _ _ _ _) (fun _ =>
      safe_bind _ _ (safe_activation _ _) (fun _ => safe_emit _ trivial))
  · exact safe_emit _ trivial

def Allowed (output : Tensor) (point : Address) : Prop :=
  point.1 = output.storage ∧ output.offset ≤ point.2 ∧ point.2-output.offset < output.rows*output.columns

theorem safe_atomic (p : Problem) (b : Buffers) (layout : Layout p b) (tile index : Nat) :
    Code.Safe (Allowed b.C) (atomicCode (α:=α) p b tile index) := by
  unfold atomicCode
  dsimp only
  split
  · apply safe_bind _ _ (safe_value _ _ _ _)
    intro value
    apply safe_emit
    have bound := written_offset p b layout tile _ ‹Source.writes p b tile _›
    exact ⟨rfl,bound.1,by simpa only [layout.cShape.1,layout.cShape.2] using bound.2⟩
  · trivial

theorem safe_sequence (f : Nat → Code α Unit) (items : List Nat)
    (h : ∀ i ∈ items, Code.Safe allowed (f i)) : Code.Safe allowed (sequenceCode f items) := by
  induction items with
  | nil => trivial
  | cons i rest ih => exact safe_bind _ _ (h i (by simp)) (fun _ => ih (fun j hj => h j (by simp [hj])))

theorem safe_grid (p : Problem) (b : Buffers) (layout : Layout p b) : Code.Safe (Allowed b.C) (gridCode (α:=α) p b) :=
  safe_sequence _ _ (fun _ _ => safe_sequence _ _ (fun _ _ => safe_atomic p b layout _ _))

theorem safe_zero (output : Tensor) : Code.Safe (Allowed output) (zeroCode (α:=α) output) := by
  apply safe_sequence
  intro i hi
  apply safe_bind _ _ (safe_emit (.zero : Event α α) trivial)
  intro zero
  apply safe_emit
  have bound := List.mem_range.mp hi
  exact ⟨rfl,by omega,by omega⟩

theorem safe_replay (p : Problem) (b : Buffers) (layout : Layout p b) (count : Nat) :
    Code.Safe (Allowed b.C) (replayCode (α:=α) p b count) := by
  induction count with
  | zero => trivial
  | succ count ih => exact safe_bind _ _ (safe_grid p b layout) (fun _ => ih)

theorem safe_computation (args : Host.Arguments) (output : Tensor) (layout : Layout args.problem ⟨args.A,args.B,output⟩)
    (trials : Nat) : Code.Safe (Allowed output) (computationCode (α:=α) args output trials) := by
  unfold computationCode
  split_ifs
  · trivial
  · exact safe_zero output
  · exact safe_replay _ _ layout _

#axiomsClean safe_computation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
