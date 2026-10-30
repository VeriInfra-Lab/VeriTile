import bench.repairs.tritonbench_g.embedding_triton_kernel.IO
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Formula
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.ProgramSafety
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.IOImplements
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

private def atProgram (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0}

private theorem input_ids (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (xs : ChannelValues α (IO.inputChannels p)) (s : State α)
    (input : (IO.embeddingIO rest zero candidate p pid).InputBinding xs s)
    (a : Fin 64) (active : IO.active p pid a) :
    (s.mem p.ids (pid*64+a.val)).read .int = IO.ids p xs a :=
  input (0 : Fin 2) (a,PUnit.unit) active

private theorem input_weight (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (xs : ChannelValues α (IO.inputChannels p)) (s : State α)
    (input : (IO.embeddingIO rest zero candidate p pid).InputBinding xs s)
    (k : Fin (IO.vocabulary p)) (d : Fin p.width)
    (selected : IO.selected p pid (IO.ids p xs) k) (active : d.val<p.hidden) :
    (s.mem p.weight (k.val*p.weightStride+d.val)).read .real = IO.weights p xs k d :=
  input (1 : Fin 2) (k,d,PUnit.unit) ⟨selected,active⟩

/-- Derive each actual source address obligation from the shared IO bounds,
using the logical metadata binding for the data-dependent weight addresses. -/
private theorem access_contract (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (layout : Execution.Layout p)
    (xs : ChannelValues α (IO.inputChannels p)) (bounds : RegionBounds)
    (bound : (IO.embeddingIO rest zero candidate p pid).InBounds bounds xs)
    (s : State α) (input : (IO.embeddingIO rest zero candidate p pid).InputBinding xs s) :
    StructuralAccess.AccessContract p ((atProgram pid s).pids 0)
      (fun r => ((atProgram pid s).mem p.ids r).read .int) bounds := by
  change StructuralAccess.AccessContract p pid (fun r => (s.mem p.ids r).read .int) bounds
  refine ⟨layout,?_,?_,?_⟩
  · intro a ha active
    exact bound.2.1 (0 : Fin 2) (⟨a,ha⟩,PUnit.unit) active
  · intro a ha active lo hi d hd
    let lane : Fin 64 := ⟨a,ha⟩
    have token := input_ids rest zero candidate p pid xs s input lane active
    rw [token] at lo hi ⊢
    let k : Fin (IO.vocabulary p) := ⟨(IO.ids p xs lane-p.first).toNat,by unfold IO.vocabulary; omega⟩
    let col : Fin p.width := ⟨d,lt_of_lt_of_le hd layout.width_covers⟩
    have selected : IO.selected p pid (IO.ids p xs) k := ⟨lane,active,lo,hi,rfl⟩
    exact bound.2.1 (1 : Fin 2) (k,col,PUnit.unit) ⟨selected,hd⟩
  · intro a ha active d hd
    exact bound.2.2.1 (0 : Fin 1)
      (⟨a,ha⟩,⟨d,lt_of_lt_of_le hd layout.width_covers⟩,PUnit.unit) ⟨active,hd⟩

/-- The selected scalar payload is the independent vocabulary-table formula. -/
private theorem value_formula (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (xs : ChannelValues α (IO.inputChannels p)) (s : State α)
    (input : (IO.embeddingIO rest zero candidate p pid).InputBinding xs s)
    (a : Fin 64) (d : Fin p.width) (active : IO.active p pid a) (col : d.val<p.hidden) :
    Execution.value zero p (atProgram pid s) a.val d.val =
      Mathematics.formula zero p (IO.ids p xs) (IO.weights p xs) a d := by
  have token := input_ids rest zero candidate p pid xs s input a active
  change (if pid*64+a.val<p.n ∧ p.first ≤ (s.mem p.ids (pid*64+a.val)).read .int ∧
      (s.mem p.ids (pid*64+a.val)).read .int<p.last then
      (s.mem p.weight (((s.mem p.ids (pid*64+a.val)).read .int-p.first).toNat*p.weightStride+d.val)).read .real
    else zero) = _
  rw [token]
  simp only [show pid*64+a.val<p.n from active,true_and]
  by_cases valid : p.first ≤ IO.ids p xs a ∧ IO.ids p xs a<p.last
  · rw [if_pos valid,Mathematics.formula,dif_pos valid]
    exact input_weight rest zero candidate p pid xs s input
      ⟨(IO.ids p xs a-p.first).toNat,by unfold IO.vocabulary; omega⟩ d
      ⟨a,active,valid.1,valid.2,rfl⟩ col
  · simp only [if_neg valid,Mathematics.formula,dif_neg valid]

/-- Complete source/candidate IO; execution and safety follow the actual source,
while the result formula reads the original logical weight table. -/
theorem implements (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (layout : Execution.Layout p) :
    (IO.embeddingIO rest zero candidate p pid).Implements
      (IO.outputValues p (Mathematics.formula zero p)) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bound s input
    have access := access_contract rest zero candidate p pid layout xs bounds bound s input
    cases candidate
    · have safe := StructuralAccess.source_safe_of_access rest zero p bounds (atProgram pid s) access
      simpa [IO.embeddingIO,ScheduleSafe,Invocation.Safe,Invocation.initial,bindRegisters,atProgram] using safe
    · have safe := StructuralAccess.candidate_safe_of_access rest zero p bounds (atProgram pid s) access
      simpa [IO.embeddingIO,ScheduleSafe,Invocation.Safe,Invocation.initial,bindRegisters,atProgram] using safe
  · intro xs s input
    obtain ⟨t,ran,_,values,frame⟩ := Execution.program_correct rest zero .int32 p layout (atProgram pid s)
    refine ⟨t,?_,?_,?_⟩
    · cases candidate
      · simpa [IO.embeddingIO,runSchedule,Invocation.run,Invocation.initial,bindRegisters,atProgram] using ran
      · have candidateRun := (Rewrite.execution_eq (Execution.backend rest zero) p (atProgram pid s)).symm.trans ran
        simpa [IO.embeddingIO,runSchedule,Invocation.run,Invocation.initial,bindRegisters,atProgram] using candidateRun
    · intro output j live
      have houtput : output=(0 : Fin 1) := @Subsingleton.elim (Fin 1) inferInstance output 0
      subst output
      rcases j with ⟨a,d,u⟩
      cases u
      change IO.active p pid a ∧ d.val<p.hidden at live
      change mapCell id (t.mem p.output ((pid*64+a.val)*p.outputStride+d.val))=
        .mk .real (Mathematics.formula zero p (IO.ids p xs) (IO.weights p xs) a d)
      have stored := values a.val a.isLt live.1 d.val live.2
      change t.mem p.output ((pid*64+a.val)*p.outputStride+d.val)=_ at stored
      rw [stored]
      change Cell.mk .real (Execution.value zero p (atProgram pid s) a.val d.val)=_
      rw [value_formula rest zero candidate p pid xs s input a d live.1 live.2]
    · intro r off outside
      apply frame r off
      by_cases same : r=p.output
      · right
        intro a ha active d hd
        let lane : Fin 64 := ⟨a,ha⟩
        let col : Fin p.width := ⟨d,lt_of_lt_of_le hd layout.width_covers⟩
        have miss := outside.1 (0 : Fin 1) (lane,col,PUnit.unit) ⟨active,hd⟩
        exact miss.resolve_left (fun wrong => wrong same)
      · exact Or.inl same

/-- Public source domain retains signed int32 IDs; payloads remain unrestricted. -/
theorem implements_signed (rest : Algebra α) (zero : α) (candidate : Bool)
    (p : Parameters) (pid : Nat) (layout : Execution.Layout p) :
    (IO.embeddingIO rest zero candidate p pid).ImplementsOn (IO.signed p pid)
      (IO.outputValues p (Mathematics.formula zero p)) := by
  intro xs _ bounds bound s input
  exact implements rest zero candidate p pid layout xs trivial bounds bound s input

#axiomsClean implements
#axiomsClean implements_signed
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.IOImplements
