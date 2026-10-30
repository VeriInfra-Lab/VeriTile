import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.LogicalGrid
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host RealSpec
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem input_reads (rest : Algebra Word) (h : Input) (jobs : List Nat)
    (encode : A → Word) (observe : Word → A) (values : Inputs A h) (s : State Word)
    (bound : (conversionIO rest h jobs encode observe).InputBinding values s) : Reads h values s := by
  intro i
  exact bound ⟨0,by simp [conversionIO,memory,inputChannels]⟩ (i,PUnit.unit) trivial

theorem invocation_safe (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (encode : A → Word) (observe : Word → A) (values : Inputs A h)
    (bounds : RegionName → Nat)
    (backed : (conversionIO rest h jobs encode observe).InBounds bounds values)
    (pid : Nat) (gridSafe : grid h≤2147483647) (valid : pid<grid h) (s : State Word) :
    (invocation rest h pid).Safe bounds s := by
  change TraceSafe (backend rest) bounds (Kernels.source (Launch.parameters h))
    (programState h ((BitVec.ofNat 32 pid).signExtend 64).toNat s)
  have count := Launch.count_exact h storage
  have pidExact := Launch.pid_exact h pid gridSafe valid
  unfold Launch.parameters
  rw [count,pidExact]
  apply Access.source_safe
  · intro i live
    exact backed.2.1 ⟨0,by simp [conversionIO,memory,inputChannels]⟩
      (⟨pid*512+i.val,live⟩,PUnit.unit) trivial
  · intro j live
    exact backed.2.2.1 ⟨0,by simp [conversionIO,memory,outputChannels]⟩
      (⟨pid*1024+j.1.val,live⟩,PUnit.unit) trivial

theorem schedule_safe (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (encode : A → Word) (observe : Word → A) (values : Inputs A h)
    (bounds : RegionName → Nat)
    (backed : (conversionIO rest h jobs encode observe).InBounds bounds values)
    (gridSafe : grid h≤2147483647) (valid : ∀ pid∈jobs,pid<grid h) (s : State Word) :
    ScheduleSafe bounds (jobs.map (invocation rest h)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    constructor
    · exact invocation_safe rest h storage (pid::jobs) encode observe values bounds backed
        pid gridSafe (valid pid (by simp)) s
    · intro t _
      exact ih backed (fun p member => valid p (by simp [member])) t

/-- Shared complete IO assembly for either native words or Real observations.
The target is a logical array; actual-source access safety remains a conclusion. -/
theorem implements (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) (encode : A → Word) (observe : Word → A)
    (target : Inputs A h → Outputs A h)
    (formula : ∀ values (j : Fin (h.shape.count*2)),
      observe (embed ⟨Bits.expectedBits (nibble values j)⟩)=
        target values ⟨0,by simp [outputChannels]⟩ (j,PUnit.unit)) :
    (conversionIO rest h jobs encode observe).ImplementsOn (ByteDomain h) target := by
  intro values domain bounds backed s bound
  have reads := input_reads rest h jobs encode observe values s bound
  obtain ⟨t,ran,memory⟩ := complete_result rest h values domain storage jobs s fresh reads gridSafe order
  refine ⟨schedule_safe rest h storage jobs encode observe values bounds backed gridSafe
    (fun pid member => List.mem_range.mp (order.mem_iff.mp member)) s,t,
    (schedule_run rest h jobs s).trans ran,?_,?_⟩
  · intro o j _
    fin_cases o
    change mapCell observe (t.mem h.output j.1.val)=
      Cell.mk .bf16 (target values ⟨0,by simp [outputChannels]⟩ j)
    rw [memory]
    have cell : targetMemory h values s h.output j.1.val=
        Cell.mk .bf16 (embed ⟨Bits.expectedBits (nibble values j.1)⟩) := by
      simp only [targetMemory,j.1.isLt,and_self,dite_true]
    rw [cell]
    exact congrArg (Cell.mk .bf16) (formula values j.1)
  · intro r offset outside
    rw [memory]
    have inactive : ¬(r=h.output ∧ offset<h.shape.count*2) := by
      rintro ⟨hr,hn⟩
      have missed := outside.1 ⟨0,by simp [conversionIO,ConversionIO.memory,outputChannels]⟩
        (⟨offset,hn⟩,PUnit.unit) trivial
      exact missed.elim (fun ne => ne hr) (fun ne => ne rfl)
    simp only [targetMemory,inactive,dite_false]

theorem native_implements (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    (nativeIO rest h jobs).ImplementsOn (ByteDomain h) (nativeTarget h) :=
  implements rest h storage jobs fresh gridSafe order id id (nativeTarget h) (fun _ _ => rfl)

theorem real_implements (rest : Algebra Word) (h : Input) (storage : Finite.Storage h.shape.count)
    (jobs : List Nat) (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (order : jobs.Perm (List.range (grid h))) :
    (realIO rest h jobs).ImplementsOn (ByteDomain h) (mathematical h) := by
  apply implements rest h storage jobs fresh gridSafe order
  intro values j
  change realObservation (embed ⟨Bits.expectedBits (nibble values j)⟩)=some (e2m1 (nibble values j))
  have narrow : (embed ⟨Bits.expectedBits (nibble values j)⟩).bits.truncate 16=
      Bits.expectedBits (nibble values j) := by
    simp only [embed,BitVec.setWidth_eq]
    exact BitVec.setWidth_setWidth_of_le _ (by decide : 16≤32)
  simp only [realObservation,narrow,expected_real]

#axiomsClean native_implements
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
