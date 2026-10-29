import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperTensorRegions

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def namedOptional (region : RegionName) (value : Option Tensor) : List (RegionName × Tensor) :=
  (optionTensors value).map (fun tensor => (region,tensor))

def inputViews (input : Inputs) : List (RegionName × Tensor) :=
  [(preparedRegions.x,input.x),(preparedRegions.weight,input.weight)]++
    namedOptional preparedRegions.bias input.bias++namedOptional preparedRegions.residual input.residual++
    namedOptional preparedRegions.x1 input.x1++namedOptional preparedRegions.weight1 input.weight1++
    namedOptional preparedRegions.bias1 input.bias1++namedOptional preparedRegions.rowscale input.rowscale

def outputViews (output : Outputs) : List (RegionName × Tensor) :=
  [(preparedRegions.y,output.y)]++namedOptional preparedRegions.y1 output.y1++
    namedOptional preparedRegions.residualOut output.residualOut++namedOptional preparedRegions.mean output.mean++
    [(preparedRegions.rstd,output.rstd)]++namedOptional preparedRegions.seeds output.seeds++
    namedOptional preparedRegions.dropoutMask output.mask

theorem outputViews_values (output : Outputs) : (outputViews output).map Prod.snd=outputTensors output := by
  simp [outputViews,outputTensors,namedOptional,List.map_map,Function.comp_def]

theorem namedOptional_member {region : RegionName} {value : Option Tensor} {r : RegionName} {tensor : Tensor}
    (h : (r,tensor)∈namedOptional region value) : value=some tensor := by
  cases value with
  | none => simp [namedOptional,optionTensors] at h
  | some t =>
    have he : r=region ∧ tensor=t := by simpa [namedOptional,optionTensors] using h
    rw [he.2]

theorem regionTensor_member (p : Prepared) (region : RegionName) (tensor : Tensor)
    (h : regionTensor p region=some tensor) :
    (region,tensor)∈inputViews p.inputs ∨ (region,tensor)∈outputViews p.outputs := by
  cases region with
  | mk name =>
    unfold regionTensor at h
    split at h
    all_goals simp_all [inputViews,outputViews,namedOptional,optionTensors,preparedRegions]

theorem option_present_member {S Value : Type} {store : TensorStore S Descriptor Value}
    {s : S} {value : Option Tensor} (h : OptionPresent store s value)
    (region r : RegionName) (tensor : Tensor) (hm : (r,tensor)∈namedOptional region value) :
    store.present s tensor := by
  have he := namedOptional_member hm
  simpa only [OptionPresent,he] using h

theorem inputViews_present {S Value : Type} {store : TensorStore S Descriptor Value}
    {q : Request} {initial current : S} {input : Inputs}
    (h : InputsMaterialized store q initial input current)
    (region : RegionName) (tensor : Tensor) (hm : (region,tensor)∈inputViews input) :
    store.present current tensor := by
  simp only [inputViews,List.mem_append,List.mem_cons,List.not_mem_nil,or_false,or_assoc] at hm
  rcases hm with hm | hm | hm | hm | hm | hm | hm | hm
  · obtain ⟨_,he⟩ := Prod.mk.inj hm
    rw [he]
    exact h.x.present
  · obtain ⟨_,he⟩ := Prod.mk.inj hm
    rw [he]
    exact h.weight.present
  · exact option_present_member (optional_materialized_present h.bias) _ region tensor hm
  · exact option_present_member (optional_materialized_present h.residual) _ region tensor hm
  · exact option_present_member (optional_materialized_present h.x1) _ region tensor hm
  · exact option_present_member (optional_materialized_present h.weight1) _ region tensor hm
  · exact option_present_member (optional_materialized_present h.bias1) _ region tensor hm
  · exact option_present_member (optional_materialized_present h.rowscale) _ region tensor hm

theorem pairwise_symmetric_distinct {β : Type} {R : β → β → Prop}
    (hs : ∀ a b, R a b → R b a) (xs : List β) (hp : xs.Pairwise R)
    (a b : β) (ha : a∈xs) (hb : b∈xs) (hne : a≠b) : R a b := by
  induction xs with
  | nil => simp at ha
  | cons first rest ih =>
    obtain ⟨hfirst,hrest⟩ := List.pairwise_cons.mp hp
    rcases List.mem_cons.mp ha with ha | ha
    · subst a
      rcases List.mem_cons.mp hb with hb | hb
      · exact False.elim (hne hb.symm)
      · exact hfirst b hb
    · rcases List.mem_cons.mp hb with hb | hb
      · subst b
        exact hs _ _ (hfirst a ha)
      · exact ih hrest ha hb

/-- Byte separation is inherited from the actual sequential allocation
batch. Distinct source output tags remain separate even for empty tensors. -/
theorem outputViews_separate {S Value : Type} {store : TensorStore S Descriptor Value}
    {initial current : S} {output : Outputs}
    (h : AllocationBatch store initial (outputTensors output) current)
    (ra rb : RegionName) (a b : Tensor)
    (ha : (ra,a)∈outputViews output) (hb : (rb,b)∈outputViews output) (hne : ra≠rb) :
    Separate store current a b := by
  have hp : (outputViews output).Pairwise (fun a b => Separate store current a.2 b.2) := by
    apply List.pairwise_map.mp
    rw [outputViews_values]
    exact h.pairwise
  exact pairwise_symmetric_distinct (fun _ _ h => h.symm) (outputViews output) hp (ra,a) (rb,b) ha hb
    (fun he => hne (congrArg Prod.fst he))

#axiomsClean outputViews_values
#axiomsClean namedOptional_member
#axiomsClean regionTensor_member
#axiomsClean inputViews_present
#axiomsClean pairwise_symmetric_distinct
#axiomsClean outputViews_separate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
