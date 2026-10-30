import bench.tritonbench_g.masked_select.Kernels
import bench.tritonbench_g.masked_select.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedSelect.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect
open TypedCopy Whole StableSelection
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section

/-- Independent mathematical output: stable selection in input index order. -/
def mathematical {A : Type} (input : Nat → A) (mask : Nat → Bool) (n : Nat) : List A :=
  ((List.range n).filter mask).map input

def Frame {A : Type} (p : Parameters) (mask : Nat → Bool) (before after : State A) : Prop :=
  ∀ r o, r ≠ p.output ∨ count mask p.size ≤ o → after.mem r o = before.mem r o

def Result {A : Type} (p : Parameters) (input : Nat → A) (mask : Nat → Bool)
    (before after : State A) : Prop :=
  (∀ k, k < (mathematical input mask p.size).length →
    some (after.mem p.output k) = ((mathematical input mask p.size)[k]?).map (Cell.mk .real)) ∧
  Frame p mask before after

/-- One full source program writes exactly its selected active elements to
their stable ranks; every cell outside those written destinations is framed.
No injectivity premise is assumed: it follows from the inclusive prefix input. -/
theorem program_correct {A : Type} [Inhabited A] (rest : Algebra A) (p : Parameters)
    (input : Nat → A) (mask : Nat → Bool) (before : State A)
    (h : KernelInputs p before input mask) :
    ∃ after, FP.Structural.exec (copyBackend rest) (kernel p) before = some after ∧
      (∀ lane : TileIndex [p.width], mask (inputOffset p before lane) = Bool.true →
        inputOffset p before lane < p.size →
        after.mem p.output (count mask (inputOffset p before lane)) =
          .mk .real (input (inputOffset p before lane))) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ lane : TileIndex [p.width],
        mask (inputOffset p before lane) = Bool.true → inputOffset p before lane < p.size →
        o ≠ count mask (inputOffset p before lane)) → after.mem r o = before.mem r o) := by
  refine ⟨(TypedCopy.execution rest .fp32 p before).val,
    (TypedCopy.execution rest .fp32 p before).property,?_,?_⟩
  · intro lane hm hi
    have active := (active_iff p before input mask h lane).mpr ⟨hm,hi⟩
    rw [← offset_selected p before input mask h lane active]
    exact TypedCopy.typed_readback rest .fp32 p before lane active
      (prefix_injective p before input mask h) _ (h.payload _ hi)
  · intro r o outside
    apply TypedCopy.frame
    exact outside.elim Or.inl (fun hmiss => Or.inr (fun lane active => by
      have ha := (active_iff p before input mask h lane).mp active
      rw [offset_selected p before input mask h lane active]
      exact hmiss lane ha.1 ha.2))

/-- Already established kernel-grid composition, specialized to an actual
autotune configuration. No wrapper, producer or autotuner execution is modeled. -/
def run {A : Type} [Inhabited A] (rest : Algebra A) (p : Parameters)
    (config : Config) (before : State A) : Option (State A) :=
  runGrid rest {p with width := config.width}
    (List.range (grid p.size config.width)) before

theorem complete {A : Type} [Inhabited A] (rest : Algebra A) (p : Parameters)
    (config : Config) (hc : config ∈ configs) (input : Nat → A) (mask : Nat → Bool)
    (before : State A) (h : KernelInputs p before input mask) (sep : KernelSeparate p) :
    ∃ after, run rest p config before = some after ∧ Result p input mask before after := by
  obtain ⟨after,executed,_,values,frame⟩ := complete_grid rest {p with width := config.width}
    (List.range (grid p.size config.width)) before input mask
    ⟨h.payload,h.selection,h.cumulative⟩ sep (config_width_pos config hc)
    (fun pid hp => List.mem_range.mpr hp)
  exact ⟨after,executed,values,frame⟩

/-- Any two complete runs of actual source configurations have the same
entire memory, by the independent stable result and exterior frame. -/
theorem configurations_equivalent {A : Type} [Inhabited A] (rest : Algebra A) (p : Parameters)
    (left right : Config) (hl : left ∈ configs) (hr : right ∈ configs)
    (input : Nat → A) (mask : Nat → Bool) (before : State A)
    (h : KernelInputs p before input mask) (sep : KernelSeparate p) :
    ∃ a b, run rest p left before = some a ∧ run rest p right before = some b ∧
      a.mem = b.mem ∧ Frame p mask before a := by
  obtain ⟨a,ha,va,fa⟩ := complete rest p left hl input mask before h sep
  obtain ⟨b,hb,vb,fb⟩ := complete rest p right hr input mask before h sep
  refine ⟨a,b,ha,hb,?_,fa⟩
  funext r o
  by_cases output : r = p.output
  · subst r
    by_cases inside : o < count mask p.size
    · have length : o < (mathematical input mask p.size).length := by
        change o < (StableSelection.select input mask p.size).length
        simpa only [length_select] using inside
      exact Option.some.inj ((va o length).trans (vb o length).symm)
    · exact (fa _ _ (Or.inr (by omega))).trans (fb _ _ (Or.inr (by omega))).symm
  · exact (fa _ _ (Or.inl output)).trans (fb _ _ (Or.inl output)).symm

#specNonCircular mathematical avoiding [kernel,run]
#axiomsClean program_correct
#axiomsClean complete
#axiomsClean configurations_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.Kernel
