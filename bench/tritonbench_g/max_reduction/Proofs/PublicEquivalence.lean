import bench.tritonbench_g.max_reduction.Proofs.AccessSafety
import bench.tritonbench_g.max_reduction.Proofs.Stores
import bench.tritonbench_g.max_reduction.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.MaxReduction.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem stage_safe (e : Source.Engine α) (second candidate : Bool) (inp out : RegionName)
    (N B pid : Nat) (hB : 0<B) (xs : (stageIO e second inp out N B pid).Inputs)
    (bounds : RegionName→Nat) (covered : (stageIO e second inp out N B pid).InBounds bounds xs) (s : State α) :
    ScheduleSafe bounds [stageLaunch e second candidate inp out N B pid] s := by
  constructor
  · apply AccessSafety.stage e second candidate inp out N B hB
    · intro j hj
      exact covered.2.1 (0 : Fin 1) (j,PUnit.unit) hj
    · exact covered.2.2.1 (0 : Fin 1) PUnit.unit trivial
  · intro _ _
    trivial

theorem stage_equivalent (e : Source.Engine α) (second : Bool) (inp out : RegionName)
    (N B pid : Nat) (hB : 0<B) :
    (stageIO e second inp out N B pid).Equivalent (stageCandidate e second inp out N B pid) := by
  intro xs _ bounds covered s _
  refine ⟨stage_safe e second Bool.false inp out N B pid hB xs bounds covered s,
    stage_safe e second Bool.true inp out N B pid hB xs bounds covered s,?_⟩
  let entry := (stageLaunch e second Bool.false inp out N B pid).initial s
  have missed (r : RegionName) (o : Nat) (h : (stageIO e second inp out N B pid).Outside xs r o) :
      r≠out ∨ o≠stageOutput second pid := h.1 (0 : Fin 1) PUnit.unit trivial
  cases second with
  | «false» =>
    obtain ⟨final,original,candidate,memory⟩ := RawFP.stage1_success e inp out N B hB entry
    refine ⟨final,final,?_,?_,?_,?_,?_⟩
    · simpa [stageIO,stageLaunch,stageSource,runSchedule,Invocation.run] using original
    · simpa [stageCandidate,stageLaunch,stageSource,runSchedule,Invocation.run] using candidate
    · intro _ _ _
      rfl
    all_goals
      intro r o outside
      rw [memory]
      exact State.write_other entry out r pid o _ (missed r o outside)
  | «true» =>
    obtain ⟨left,right,original,candidate,same,memory⟩ := RawFP.stage2_success e inp out N B hB entry
    refine ⟨left,right,?_,?_,?_,?_,?_⟩
    · simpa [stageIO,stageLaunch,stageSource,runSchedule,Invocation.run] using original
    · simpa [stageCandidate,stageLaunch,stageSource,runSchedule,Invocation.run] using candidate
    · intro _ _ _
      simp only [same]
      rfl
    · intro r o outside
      rw [memory]
      exact State.write_other entry out r 0 o _ (missed r o outside)
    · intro r o outside
      rw [←same,memory]
      exact State.write_other entry out r 0 o _ (missed r o outside)

theorem dimension_safe {γ : Type} (e : Source.Engine α) (encode : γ→α) (observe : α→γ)
    (candidate : Bool) (inp ov oi : RegionName) (M N K BM BN p k : Nat) (hBN : 0<BN)
    (xs : (dimensionIO e encode observe inp ov oi M N K BM BN p k).Inputs)
    (bounds : RegionName→Nat)
    (covered : (dimensionIO e encode observe inp ov oi M N K BM BN p k).InBounds bounds xs) (s : State α) :
    ScheduleSafe bounds [dimensionLaunch e candidate inp ov oi M N K BM BN p k] s := by
  constructor
  · apply AccessSafety.dimension e candidate inp ov oi M N K BM BN hBN
    · intro i j hm hn
      exact covered.2.1 (0 : Fin 1) (i,j,PUnit.unit) ⟨hm,hn⟩
    · intro i hm
      exact covered.2.2.1 (0 : Fin 2) (i,PUnit.unit) hm
    · intro i hm
      exact covered.2.2.1 (1 : Fin 2) (i,PUnit.unit) hm
  · intro _ _
    trivial

theorem dimension_frame {γ : Type} (e : Source.Engine α) (encode : γ→α) (observe : α→γ)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat)
    (xs : (dimensionIO e encode observe inp ov oi M N K BM BN p k).Inputs)
    (s : State α) (r : RegionName) (o : Nat)
    (outside : (dimensionIO e encode observe inp ov oi M N K BM BN p k).Outside xs r o) :
    (Source.dimensionStores e inp ov oi M N K BM BN
      ((dimensionLaunch e Bool.false inp ov oi M N K BM BN p k).initial s)).mem r o=s.mem r o := by
  apply Stores.frame
  · by_cases neq : r≠ov
    · exact Or.inl neq
    · exact Or.inr (fun i active => (outside.1 (0 : Fin 2) i active).resolve_left neq)
  · by_cases neq : r≠oi
    · exact Or.inl neq
    · exact Or.inr (fun i active => (outside.1 (1 : Fin 2) i active).resolve_left neq)

theorem dimension_equivalent (e : Source.Engine α) (inp ov oi : RegionName)
    (M N K BM BN p k : Nat) (hBN : 0<BN) :
    (dimensionIO e id id inp ov oi M N K BM BN p k).Equivalent
      (dimensionCandidate e inp ov oi M N K BM BN p k) := by
  intro xs _ bounds covered s _
  refine ⟨dimension_safe e id id Bool.false inp ov oi M N K BM BN p k hBN xs bounds covered s,
    dimension_safe e id id Bool.true inp ov oi M N K BM BN p k hBN xs bounds covered s,?_⟩
  obtain ⟨final,original,candidate,memory⟩ := RawFP.dimension_success e inp ov oi M N K BM BN hBN
    ((dimensionLaunch e Bool.false inp ov oi M N K BM BN p k).initial s)
  refine ⟨final,final,?_,?_,?_,?_,?_⟩
  · simpa [dimensionIO,dimensionLaunch,runSchedule,Invocation.run] using original
  · simpa [dimensionCandidate,dimensionLaunch,runSchedule,Invocation.run] using candidate
  · intro _ _ _
    rfl
  all_goals
    intro r o outside
    rw [memory]
    exact dimension_frame e id id inp ov oi M N K BM BN p k xs s r o outside

#axiomsClean stage_equivalent
#axiomsClean dimension_equivalent
end
end VeriTile.Bench.Repairs.MaxReduction.PublicIO
