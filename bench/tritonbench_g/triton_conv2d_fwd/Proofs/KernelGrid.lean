import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.Conv2d.KernelGrid
open VeriTile.Triton FP.Structural KernelBinding
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 8192
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def backend (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α) (f : Flags) :=
  if f.fp16 then KernelHalf.backend M H else KernelDefault.backend M D

def value (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (pid : Program) (t : State α) : Index → α :=
  if f.fp16 then KernelHalf.outputValue M H (parameters s f) (atProgram pid t)
  else KernelDefault.outputValue M D (parameters s f) (atProgram pid t)

def owner (x : Conv2d.Cell) : Program := (x.spatial/128,x.feature/32,x.group)
def index (x : Conv2d.Cell) : Index :=
  (⟨x.spatial%128,Nat.mod_lt _ (by decide)⟩,⟨x.feature%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit)

def point (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (t : State α) (x : Conv2d.Cell) : α :=
  value M H D s f (owner x) t (index x)

@[simp] theorem cell_index (x : Conv2d.Cell) : cell (owner x) (index x)=x := by
  cases x
  simp only [cell,cellOf,lane,owner,index,Conv2d.Cell.mk.injEq,true_and]
  constructor <;> omega

@[simp] theorem owner_cell (pid : Program) (i : Index) : owner (cell pid i)=pid := by
  have a:=i.1.isLt
  have b:=i.2.1.isLt
  apply Prod.ext
  · dsimp [owner,cell,cellOf,lane]; omega
  · apply Prod.ext
    · dsimp [owner,cell,cellOf,lane]; omega
    · rfl

@[simp] theorem index_cell (pid : Program) (i : Index) : index (cell pid i)=i := by
  apply KernelBinding.index_injective
  have a:=i.1.isLt
  have b:=i.2.1.isLt
  apply Prod.ext <;> dsimp [index,cell,cellOf,lane] <;> omega

@[simp] theorem point_cell (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (t : State α) (pid : Program) (i : Index) :
    point M H D s f t (cell pid i)=value M H D s f pid t i := by
  simp [point]

def SameInputs (a b : State α) : Prop :=
  (∀ o,a.mem "input" o=b.mem "input" o) ∧ (∀ o,a.mem "weight" o=b.mem "weight" o)

theorem value_inputs (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (pid : Program) (a b : State α) (eq : SameInputs a b) :
    value M H D s f pid a=value M H D s f pid b := by
  funext i
  unfold value
  split
  · unfold KernelHalf.outputValue KernelHalf.accumulation KernelHalf.widths KernelHalf.channels KernelHalf.update KernelHalf.inputValue KernelHalf.weightValue
    simp only [KernelHalf.inputMask, KernelHalf.weightMask, KernelHalf.inputOffset, KernelHalf.weightOffset, KernelHalf.inputBase, KernelHalf.weightBase, KernelHalf.ih, KernelHalf.iw, KernelHalf.oh, KernelHalf.ow, KernelHalf.feat, KernelHalf.batch, KernelHalf.bh, KernelHalf.bhw, atProgram,parameters,eq.1,eq.2]
  · unfold KernelDefault.outputValue KernelDefault.accumulation KernelDefault.widths KernelDefault.channels KernelDefault.update KernelDefault.inputValue KernelDefault.weightValue
    simp only [KernelDefault.inputMask, KernelDefault.weightMask, KernelDefault.inputOffset, KernelDefault.weightOffset, KernelDefault.inputBase, KernelDefault.weightBase, KernelDefault.ih, KernelDefault.iw, KernelDefault.oh, KernelDefault.ow, KernelDefault.feat, KernelDefault.batch, KernelDefault.bh, KernelDefault.bhw, atProgram,parameters,eq.1,eq.2]

theorem point_inputs (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (a b : State α) (eq : SameInputs a b) (x : Conv2d.Cell) :
    point M H D s f a x=point M H D s f b x := congrFun (value_inputs M H D s f _ a b eq) _

def program (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (pid : Program) (t : State α) : Option (State α) :=
  FP.Structural.exec (backend M H D f) (KernelSource.kernel f.fp16 (parameters s f)) (atProgram pid t)

def run (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) : List Program → State α → Option (State α)
  | [],t => some t
  | pid::ps,t => (program M H D s f pid t).bind (run M H D s f ps)

def launch (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (f : Flags) (t : State α) := run M H D s f (programs s) t

/-- One actual typed source execution, with derived store ownership and a
complete exterior frame. No execution success or output equation is assumed. -/
theorem program_correct (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (v : Valid s) (f : Flags) (pid : Program) (hg : pid.2.2<s.groups) (t : State α) :
    ∃ u,program M H D s f pid t=some u ∧
      (∀ i,s.Active (cell pid i) → u.mem "output" (outputPointer s (cell pid i))=
        .mk .fp32 (value M H D s f pid t i)) ∧
      (∀ (r : RegionName) o,(r≠"output" ∨ ∀ i,s.Active (cell pid i) → o≠outputPointer s (cell pid i)) →
        u.mem r o=t.mem r o) := by
  have hw:=stores_unique s v f pid hg t
  have conv (u : State α) (vals : Index → α)
      (post : KernelHalf.Post (parameters s f) (atProgram pid t) u vals) :
      (∀ i,s.Active (cell pid i) → u.mem "output" (outputPointer s (cell pid i))=.mk .fp32 (vals i)) ∧
      (∀ (r : RegionName) o,(r≠"output" ∨ ∀ i,s.Active (cell pid i) → o≠outputPointer s (cell pid i)) →
        u.mem r o=t.mem r o) := by
    constructor
    · intro i hi
      have h:=post.1 i ((active_iff s v f pid hg t i).mpr hi)
      rw [output_address s v f] at h
      exact h
    · intro r o h
      apply post.2
      rcases h with h|h
      · exact Or.inl h
      · exact Or.inr (fun i hi => by simpa only [output_address s v f] using h i ((active_iff s v f pid hg t i).mp hi))
  by_cases hf : f.fp16=Bool.true
  · obtain ⟨u,he,_,hp⟩:=KernelHalf.correctness M H (parameters s f) (atProgram pid t) hw
    exact ⟨u,by simpa [program,backend,KernelSource.kernel,hf] using he,
      by simpa [value,hf] using conv u _ hp⟩
  · have hf' : f.fp16=Bool.false := Bool.eq_false_iff.mpr hf
    obtain ⟨u,he,_,hp⟩:=KernelDefault.correctness M D (parameters s f) (atProgram pid t) hw
    exact ⟨u,by simpa [program,backend,KernelSource.kernel,hf'] using he,
      by simpa [value,hf'] using conv u _ hp⟩

#axiomsClean value_inputs
#axiomsClean program_correct
end
end VeriTile.Bench.Repairs.Conv2d.KernelGrid
