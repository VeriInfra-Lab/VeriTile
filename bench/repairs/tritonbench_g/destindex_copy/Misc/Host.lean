import bench.repairs.tritonbench_g.destindex_copy.Proofs.Composition

namespace VeriTile.Bench.Repairs.DestindexCopy
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def Nonempty (m : Inputs) (s : Side) : Prop :=
  0<(config m s).count ∧ 0<(config m s).heads ∧ 0<(config m s).dims

namespace Mathematical
variable {α : Type} [Inhabited α]
def effect (rest : Algebra α) (m : Inputs) (w : Nat → Word) (s : Side)
    (before : Nat → α) (other : State α) : Nat → α :=
  if Nonempty m s then DestindexCopyKv1.execute rest (plane m s) w before other else before

theorem stage (rest : Algebra α) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (s : Side) (before : Nat → α) (other : State α) :
    Stage m w s before (effect rest m w s before other) := by
  by_cases ne : Nonempty m s
  · rw [effect,if_pos ne]
    exact DestindexCopyKv1.execute_correct rest (plane m s) w (h.common s) before other
  · rw [effect,if_neg ne]
    refine ⟨⟨?_,fun _ _ => rfl⟩,fun _ _ => rfl⟩
    intro p a b hp ha hb
    exact False.elim (ne ⟨by omega,by omega,by omega⟩)

def execute (rest : Algebra α) (m : Inputs) (w : Nat → Word) (before : Nat → α)
    (other : State α) : Nat → α :=
  effect rest m w .rope (effect rest m w .nope before other) other

theorem correct (rest : Algebra α) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Nat → α) (other : State α) :
    DualCopy m w before (execute rest m w before other) ∧
    (∀ cell, Outside m w cell → execute rest m w before other cell=before cell) :=
  compose m w h before _ _ (stage rest m w h .nope before other)
    (stage rest m w h .rope _ other)
end Mathematical

namespace Physical
def effect (m : Inputs) (s : Side) (before : Memory) : Memory :=
  if Nonempty m s then
    DestindexCopyKv1.Physical.launch (plane m s) (List.range (config m s).count) before
  else before

theorem stage (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (s : Side) (before : Memory) (binding : DestindexCopyKv1.TensorBinding (plane m s) w before)
    (other : State Word) : Stage m w s before (effect m s before) := by
  by_cases ne : Nonempty m s
  · rw [effect,if_pos ne]
    exact DestindexCopyKv1.Physical.correct rest (plane m s) w (h.common s) before binding other _ (by simp [config])
  · rw [effect,if_neg ne]
    refine ⟨⟨?_,fun _ _ => rfl⟩,fun _ _ => rfl⟩
    intro p a b hp ha hb
    exact False.elim (ne ⟨by omega,by omega,by omega⟩)

def execute (m : Inputs) (before : Memory) : Memory := effect m .rope (effect m .nope before)

theorem second_binding (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before middle : Memory) (binding : TensorBinding m w before)
    (first : Stage m w .nope before middle) : DestindexCopyKv1.TensorBinding (plane m .rope) w middle := by
  refine ⟨(binding.planeBinding .rope).capacity,?_⟩
  intro p hp
  have keep := preserves m w h .nope before middle first _ _ (h.separated .nope .rope).2.1 p hp
  exact keep.trans ((binding.planeBinding .rope).metadata p hp)

theorem correct (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (before : Memory) (binding : TensorBinding m w before) (other : State Word) :
    DualCopy m w before (execute m before) ∧
    (∀ cell, Outside m w cell → execute m before cell=before cell) := by
  have first := stage rest m w h .nope before (binding.planeBinding .nope) other
  have second := stage rest m w h .rope _ (second_binding m w h before _ binding first) other
  exact compose m w h before _ _ first second
end Physical

structure Result where
  returned : Unit
  outputObject : Side → Nat
  outputMetadata : Side → VeriTile.Torch.TensorMetadata
  memory : Memory

def runWrapper (m : Inputs) (w : Nat → Word) (objects : Side → Nat) (before : Memory) : Except InputError Result := do
  checkInputs m w
  return ⟨(),objects,fun s => (plane m s) .output,Physical.execute m before⟩

theorem wrapper_correct (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (objects : Side → Nat) (before : Memory) (binding : TensorBinding m w before) (other : State Word) :
    ∃ out, runWrapper m w objects before=.ok out ∧ out.returned=() ∧ out.outputObject=objects ∧
      out.outputMetadata=(fun s => (plane m s) .output) ∧ DualCopy m w before out.memory ∧
      (∀ cell, Outside m w cell → out.memory cell=before cell) := by
  have accepted := (checkInputs_iff m w).mpr h
  exact ⟨⟨(),objects,fun s => (plane m s) .output,Physical.execute m before⟩,
    by simp [runWrapper,accepted],rfl,rfl,rfl,Physical.correct rest m w h before binding other⟩

abbrev observeByte := DestindexCopyKv1.observeByte

theorem byte_frame (m : Inputs) (w : Nat → Word) (h : InputContract m w) (before after : Memory)
    (frame : ∀ cell, Outside m w cell → after cell=before cell) (byte : Nat)
    (outside : ∀ s x, touched m w s x → byte<addr (base m s .output) x ∨ addr (base m s .output) x+4≤byte) :
    observeByte after byte=observeByte before byte := by
  unfold observeByte DestindexCopyKv1.observeByte
  rw [frame]
  intro s
  rintro ⟨x,hx,he⟩
  obtain ⟨p,a,b,hp,ha,hb,rfl⟩ := hx
  have bound := DestindexCopyKv1.output_bound (plane m s) w (h.common s) p a b hp ha hb
  have align := (h.common s).alignment .output (by omega)
  have ext := outside s _ ⟨p,a,b,hp,ha,hb,rfl⟩
  change base m s .output%4=0 at align
  unfold addr DestindexCopyKv1.addr at he ext
  omega

#axiomsClean Mathematical.correct
#axiomsClean Physical.correct
#axiomsClean wrapper_correct
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopy
