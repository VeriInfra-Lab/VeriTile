import bench.tritonbench_g.int8_quantization.Proofs.SourceExecution
import bench.tritonbench_g.int8_quantization.Proofs.SourceSchedule

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Grid
open VeriTile.Triton FP.Structural
open Schedule
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def parameters (p : Parameters) (w : Program) : Parameters := {p with base:=w.base}
def atProgram (s : State α) (w : Program) : State α :=
  {s with pids:=fun ax=> if ax=0 then w.pid else s.pids ax}

def active (p : Parameters) (w : Program) (i : Lane p) : Prop :=
  i.1.val< p.blockRows*p.channels ∧ w.index%p.blocks*p.blockRows+i.1.val/p.channels< p.length
instance (p : Parameters) (w : Program) (i : Lane p) : Decidable (active p w i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def offset (p : Parameters) (w : Program) (i : Lane p) : Nat :=
  w.index/p.blocks*p.length*p.channels+w.index%p.blocks*p.blockRows*p.channels+i.1.val

@[simp] theorem group_at (p : Parameters) (w : Program) (s : State α) :
    Model.group (parameters p w) (atProgram s w)=w.index := rfl
@[simp] theorem active_at (p : Parameters) (w : Program) (s : State α) (i : Lane p) :
    Model.active (parameters p w) (atProgram s w) i=active p w i := rfl
@[simp] theorem offset_at (p : Parameters) (w : Program) (s : State α) (i : Lane p) :
    Model.offset (parameters p w) (atProgram s w) i=offset p w i := rfl

def execute (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) :
    List Program→State α→Option (State α)
  | [],s=> some s
  | w::rest,s=> do
    let t←FP.Structural.exec (Model.backend M ops) (source (parameters p w)) (atProgram s w)
    execute M ops p rest t

def model (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width) :
    List Program→State α→State α
  | [],s=> s
  | w::rest,s=> model M ops p positive rest (finalState M ops (parameters p w) positive (atProgram s w))

theorem execute_model (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width)
    (order : List Program) (entry : State α) :
    execute M ops p order entry=some (model M ops p positive order entry) := by
  induction order generalizing entry with
  | nil=> rfl
  | cons w rest ih=> simp [execute,source_run M ops (parameters p w) positive,model,ih]

theorem model_append (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width)
    (before after : List Program) (s : State α) :
    model M ops p positive (before++after) s=
      model M ops p positive after (model M ops p positive before s) := by
  induction before generalizing s with
  | nil=> rfl
  | cons w rest ih=> simp [model,ih]

theorem frame (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width)
    (order : List Program) (entry : State α) (r : RegionName) (o : Nat)
    (scaleMiss : r≠scaleRegion ∨ ∀ w∈order,o≠w.index)
    (outputMiss : r≠outRegion ∨ ∀ w∈order,∀ i,active p w i → o≠offset p w i) :
    (model M ops p positive order entry).mem r o=entry.mem r o := by
  induction order generalizing entry with
  | nil=> rfl
  | cons w rest ih=>
    rw [model,ih _ (scaleMiss.imp id (fun h v hv=> h v (by simp [hv])))
      (outputMiss.imp id (fun h v hv=> h v (by simp [hv])))]
    exact source_frame M ops (parameters p w) positive (atProgram entry w) r o
      (scaleMiss.imp id (fun h=> h w (by simp)))
      (outputMiss.imp id (fun h=> h w (by simp)))

theorem input_preserved (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width)
    (order : List Program) (entry : State α) :
    (model M ops p positive order entry).mem xRegion=entry.mem xRegion := by
  funext o
  exact frame M ops p positive order entry xRegion o (Or.inl (by decide)) (Or.inl (by decide))

theorem values_stable (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (positive : 0< p.width)
    (w : Program) (initial current : State α) (same : current.mem xRegion=initial.mem xRegion) :
    quantized M ops (parameters p w) positive (atProgram current w)=
      quantized M ops (parameters p w) positive (atProgram initial w) ∧
    savedScale M ops (parameters p w) positive (atProgram current w)=
      savedScale M ops (parameters p w) positive (atProgram initial w) := by
  have load : loaded M (parameters p w) (atProgram current w)=loaded M (parameters p w) (atProgram initial w) := by
    funext i
    simp [loaded,atProgram,Model.active,Model.offset,Model.group,same]
  constructor
  · unfold quantized biased clipped normalized quotient denominator usable badValue scale maximum sanitized flags transformed
    simp only [load]
  · unfold savedScale badValue scale maximum sanitized flags transformed
    simp only [load]

theorem address_injective (p : Parameters) (block : 0< p.blockRows) (channels : 0< p.channels)
    (blocks : 0< p.blocks) (w v : Program) (i j : Lane p)
    (wi : active p w i) (vj : active p v j) (equal : offset p w i=offset p v j) :
    w.index=v.index ∧ i=j := by
  have inner (g lane : Nat) (row : g%p.blocks*p.blockRows+lane/p.channels< p.length) :
      g%p.blocks*p.blockRows*p.channels+lane< p.length*p.channels := by
    have rem:=Nat.mod_lt lane channels
    have decomp:=Nat.mod_add_div lane p.channels
    nlinarith
  have iw:=inner w.index i.1.val wi.2
  have iv:=inner v.index j.1.val vj.2
  have length : 0< p.length*p.channels := by omega
  have bc : 0< p.blockRows*p.channels := Nat.mul_pos block channels
  have divbatch (g lane : Nat) (h : g%p.blocks*p.blockRows*p.channels+lane< p.length*p.channels) :
      (g/p.blocks*p.length*p.channels+g%p.blocks*p.blockRows*p.channels+lane)/(p.length*p.channels)=g/p.blocks := by
    rw [Nat.add_assoc,Nat.mul_assoc,Nat.add_comm,Nat.add_mul_div_right _ _ length,Nat.div_eq_of_lt h]
    omega
  have batch : w.index/p.blocks=v.index/p.blocks := by
    have eq:=congrArg (fun n=> n/(p.length*p.channels)) equal
    simpa only [offset,divbatch _ _ iw,divbatch _ _ iv] using eq
  have tail : w.index%p.blocks*p.blockRows*p.channels+i.1.val=
      v.index%p.blocks*p.blockRows*p.channels+j.1.val := by unfold offset at equal;rw [batch] at equal;omega
  have tile : w.index%p.blocks=v.index%p.blocks := by
    have eq:=congrArg (fun n=> n/(p.blockRows*p.channels)) tail
    simp only [Nat.mul_assoc] at eq
    rw [Nat.add_comm (w.index%p.blocks*(p.blockRows*p.channels)) i.1.val,
      Nat.add_mul_div_right _ _ bc,Nat.div_eq_of_lt wi.1,
      Nat.add_comm (v.index%p.blocks*(p.blockRows*p.channels)) j.1.val,
      Nat.add_mul_div_right _ _ bc,Nat.div_eq_of_lt vj.1] at eq
    simpa using eq
  have group : w.index=v.index := by
    have hw:=Nat.mod_add_div w.index p.blocks
    have hv:=Nat.mod_add_div v.index p.blocks
    rw [batch,tile] at hw
    omega
  refine ⟨group,Prod.ext (Fin.ext ?_) (Subsingleton.elim _ _)⟩
  rw [tile] at tail
  omega

#axiomsClean execute_model
#axiomsClean frame
#axiomsClean input_preserved
#axiomsClean values_stable
#axiomsClean address_injective
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Grid
