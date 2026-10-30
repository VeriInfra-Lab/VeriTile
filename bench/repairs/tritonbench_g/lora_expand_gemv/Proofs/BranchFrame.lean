import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
attribute [local simp] VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.length VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
local instance branchFrameWord : Inhabited Word := ⟨FPValue.Word.zero⟩

theorem writeBlock_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (n : Nat) (r : RegionName) (o : Nat)
    (h : Outside p initial r o) :
    writeBlock cfg plans profile out p initial mem n r o = mem r o := by
  have aux : ∀ (xs : List (TileIndex [p.blockRows])) (acc : RegionName → Nat → Cell Word),
      xs.foldl (fun outmem i => if n+i.1.val < length p then
        fun r o => if r=p.output ∧ o=outputOffset p initial (n+i.1.val) then
          outputCell cfg plans profile out p initial mem (n+i.1.val) else outmem r o
        else outmem) acc r o = acc r o := by
    intro xs
    induction xs with
    | nil => intro acc; rfl
    | cons i xs ih =>
      intro acc
      rw [List.foldl_cons, ih]
      by_cases hi : n+i.1.val < length p
      · simp only [hi, ↓reduceIte]
        have hn : ¬ (r=p.output ∧ o=outputOffset p initial (n+i.1.val)) := by
          intro he
          rcases h with hr | ho
          · exact hr he.1
          · exact ho ⟨n+i.1.val,hi⟩ he.2
        exact if_neg hn
      · simp only [hi, ↓reduceIte]
  exact aux _ mem

theorem reference_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word)
    (fuel n : Nat) (mem : RegionName → Nat → Cell Word) (r : RegionName) (o : Nat)
    (h : Outside p initial r o) :
    reference cfg plans profile out p initial fuel n mem r o = mem r o := by
  induction fuel generalizing n mem with
  | zero => rfl
  | succ fuel ih =>
    unfold reference
    split
    · rw [ih]
      exact writeBlock_frame cfg plans profile out p initial mem n r o h
    · rfl

theorem expectedMemory_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) (r : RegionName) (o : Nat)
    (h : Outside p initial r o) :
    expectedMemory cfg plans profile out p initial r o = initial.mem r o := by
  unfold expectedMemory
  split
  · rfl
  · exact reference_frame cfg plans profile out p initial _ _ _ r o h

#axiomsClean writeBlock_frame
#axiomsClean reference_frame
#axiomsClean expectedMemory_frame
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Branches
