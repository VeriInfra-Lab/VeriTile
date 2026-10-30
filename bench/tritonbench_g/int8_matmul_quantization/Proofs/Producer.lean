import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeRun

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000

structure QuantInvocation (G K : Nat) where
  plan : Fin G → ReductionPlan K
  groups : List (Fin G)
  complete : groups.Perm (List.finRange G)

theorem quantIndex_present (v : QuantInvocation G K) (x : QuantIndex G K) :
    x∈v.groups.flatMap quantGroupIndices := by
  cases x with
  | inl x =>
    apply List.mem_flatMap.mpr
    refine ⟨x.1,v.complete.mem_iff.mpr (List.mem_finRange _),?_⟩
    simp [quantGroupIndices]
  | inr g =>
    apply List.mem_flatMap.mpr
    exact ⟨g,v.complete.mem_iff.mpr (List.mem_finRange _),by simp [quantGroupIndices]⟩

def quantRun (a : Ops I F H) (q : QuantBuffers) (v : QuantInvocation G K) (m : Memory F H) : Memory F H :=
  ((quantGridCode q v.plan v.groups).eval a m).2
theorem quantRun_eq (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (v : QuantInvocation G K) (m : Memory F H) :
    quantRun a q v m=writeAll q.writeAddress (quantExpected a q v.plan m)
      (v.groups.flatMap quantGroupIndices) m := by
  exact congrArg Prod.snd (eval_quantGridCode a q sep v.plan v.groups m m (fun _ _ => rfl))
theorem quantRun_input (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (v : QuantInvocation G K) (m : Memory F H) : QuantInputsAgree q (quantRun a q v m) m := by
  rw [quantRun_eq a q sep]
  exact quant_writeAll_input q sep _ _ m m (fun _ _ => rfl)
theorem quantRun_at (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (v : QuantInvocation G K) (m : Memory F H) (x : QuantIndex G K) :
    quantRun a q v m (q.writeAddress x)=quantExpected a q v.plan m x := by
  rw [quantRun_eq a q sep]
  exact writeAll_at _ _ (q.write_injective sep) _ m x (quantIndex_present v x)
theorem quantRun_frame (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (v : QuantInvocation G K) (m : Memory F H) (p : Address)
    (outside : ∀ x : QuantIndex G K,p≠q.writeAddress x) : quantRun a q v m p=m p := by
  rw [quantRun_eq a q sep]
  exact writeAll_frame _ _ _ m p (fun x _ => outside x)

def quantTrials (a : Ops I F H) (q : QuantBuffers) : List (QuantInvocation G K) → Memory F H → Memory F H
  | [],m => m
  | v::vs,m => quantTrials a q vs (quantRun a q v m)
theorem quantTrials_input (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (vs : List (QuantInvocation G K)) (m : Memory F H) : QuantInputsAgree q (quantTrials a q vs m) m := by
  intro p hp
  induction vs generalizing m with
  | nil => rfl
  | cons v vs ih => rw [quantTrials,ih,quantRun_input a q sep v m p hp]
theorem quantTrials_frame (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (vs : List (QuantInvocation G K)) (m : Memory F H) (p : Address)
    (outside : ∀ x : QuantIndex G K,p≠q.writeAddress x) : quantTrials a q vs m p=m p := by
  induction vs generalizing m with
  | nil => rfl
  | cons v vs ih => rw [quantTrials,ih,quantRun_frame a q sep v m p outside]

def quantWrapper (a : Ops I F H) (q : QuantBuffers)
    (trials : List (QuantInvocation G K)) (selected : QuantInvocation G K) (m : Memory F H) :=
  quantRun a q selected (quantTrials a q trials m)
theorem quantWrapper_at (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (trials : List (QuantInvocation G K)) (selected : QuantInvocation G K) (m : Memory F H)
    (x : QuantIndex G K) :
    quantWrapper a q trials selected m (q.writeAddress x)=quantExpected a q selected.plan m x := by
  unfold quantWrapper
  rw [quantRun_at a q sep]
  have same := quantTrials_input a q sep trials m
  cases x with
  | inl x => exact congrArg Sum.inl (quantValue_current a q selected.plan _ m same x)
  | inr g => simp only [quantExpected,quantScale_current a q selected.plan _ m same]
theorem quantWrapper_frame (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (trials : List (QuantInvocation G K)) (selected : QuantInvocation G K) (m : Memory F H)
    (p : Address) (outside : ∀ x : QuantIndex G K,p≠q.writeAddress x) :
    quantWrapper a q trials selected m p=m p := by
  rw [quantWrapper,quantRun_frame a q sep _ _ p outside,quantTrials_frame a q sep _ _ p outside]

def nativeQuantRun (abi : LocalEventABI I F H σ a) (q : QuantBuffers)
    (v : QuantInvocation G K) (state : σ) : σ :=
  ((quantGridCode q v.plan v.groups).native abi state).2
theorem nativeQuantRun_refines (abi : LocalEventABI I F H σ a) (q : QuantBuffers)
    (v : QuantInvocation G K) (state : σ) :
    abi.observe (nativeQuantRun abi q v state)=quantRun a q v (abi.observe state) :=
  congrArg Prod.snd ((quantGridCode q v.plan v.groups).native_refines abi state)

#axiomsClean quantRun_at
#axiomsClean quantRun_frame
#axiomsClean quantWrapper_at
#axiomsClean quantWrapper_frame
#axiomsClean nativeQuantRun_refines
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
