import bench.repairs.tritonbench_g.softmax_reducev.Proofs.RealMath

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.RealModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceV.Range (foldRange)
open FPValue
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyLogicOffset {α} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) : Nat :=
  s.pids 1*p.logicHeadStride+(sequenceStart p s+k)*p.logicSequenceStride

def keyValueOffset {α} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) (d : Fin p.dimensions) : Nat :=
  s.pids 1*p.valueHeadStride+d.val*p.valueDimStride+
    ((s.mem (Region.cast p.locations) (locationBase p s+k*p.locationSequenceStride)).read .int *
      (p.valueBatchStride : Int)).toNat

/-- Only actual input reads are related to logical real inputs. No recurrence,
reduction, output, or program-success premise appears here. Unused V memory
has no finiteness requirement. -/
structure Inputs (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) : Prop where
  logits : ∀ k, k < sequenceLength p s →
    (s.mem p.logics (keyLogicOffset p s k)).read .fp32=↑(q k)
  values : ∀ k, k < sequenceLength p s → ∀ d,
    (s.mem p.values (keyValueOffset p s k d)).read .fp32=↑(v k d)

theorem logits_closed (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v)
    (cur : Nat) (i : TileIndex [p.tokens]) :
    logits model p s cur i=qlane (sequenceLength p s) p.tokens q cur i.1 := by
  change (if cur+i.1.val < sequenceLength p s then
    (s.mem p.logics (logicOffset p s cur i.1)).read .fp32 else (⊥ : WithBot ℝ))=_
  unfold qlane
  by_cases h : cur+i.1.val < sequenceLength p s
  · rw [if_pos h,if_pos h]
    simpa only [logicOffset,keyLogicOffset,Nat.add_assoc] using input.logits (cur+i.1.val) h
  · rw [if_neg h,if_neg h]


theorem values_closed (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v)
    (cur : Nat) (i : TileIndex [p.tokens,p.dimensions]) :
    values model p s cur i=↑(if cur+i.1.val < sequenceLength p s then v (cur+i.1.val) i.2.1 else 0) := by
  change (if cur+i.1.val < sequenceLength p s then
    (s.mem p.values (valueOffset p s cur i)).read .fp32 else (↑(0.0:ℝ) : WithBot ℝ))=_
  by_cases h : cur+i.1.val < sequenceLength p s
  · rw [if_pos h,if_pos h]
    simpa only [valueOffset,keyValueOffset,location,active,if_pos h] using input.values (cur+i.1.val) h i.2.1
  · rw [if_neg h,if_neg h]
    norm_num


theorem update_eq (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v)
    (cur : Nat) (a : Accumulator (WithBot ℝ) p.dimensions) :
    update model p s cur a=step (sequenceLength p s) p.tokens p.dimensions q v cur a := by
  have eq : logits model p s cur=(fun i => qlane (sequenceLength p s) p.tokens q cur i.1) := by
    funext i
    exact logits_closed p s q v input cur i
  simp only [update,eq]
  simp_rw [values_closed p s q v input]
  rfl


theorem final_eq (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v) :
    finalAccumulator model p s=
      foldRange (step (sequenceLength p s) p.tokens p.dimensions q v) 0 (sequenceLength p s) p.tokens
        ⟨⊥,↑(0:ℝ),fun _ => ↑(0:ℝ)⟩ := by
  have eq : update model p s=step (sequenceLength p s) p.tokens p.dimensions q v := by
    funext cur a
    exact update_eq p s q v input cur a
  have init : initialAccumulator model p=⟨⊥,↑(0:ℝ),fun _ => ↑(0:ℝ)⟩ := by
    change Accumulator.mk (⊥ : WithBot ℝ) (↑(0.0 : ℝ)) (fun _ => ↑(0:ℝ))=_
    norm_num
  simp only [finalAccumulator,eq,init]


theorem source_value (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v)
    (hN : 0 < p.tokens) (hS : 0 < sequenceLength p s) (d : Index p) :
    outputValues model p (finalAccumulator model p s) d=
      ↑(target (sequenceLength p s) q (fun k => v k d.1)) := by
  rw [final_eq p s q v input]
  exact mathematical (sequenceLength p s) p.tokens p.dimensions q v hN hS d

/-- Full typed source, all dynamic loop iterations and tail masks, final
logical weighted sum, and exact frame. The only input premises concern the
input memory, legal dimensions/sequence length and injective output addresses. -/

theorem source_correct (p : Parameters) (s : State (WithBot ℝ))
    (q : Nat → ℝ) (v : Nat → Fin p.dimensions → ℝ) (input : Inputs p s q v)
    (hN : 0 < p.tokens) (hS : 0 < sequenceLength p s)
    (injective : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec model (source p) s=some final ∧
      Post p s final (fun d => ↑(target (sequenceLength p s) q (fun k => v k d.1))) := by
  obtain ⟨final,run,post⟩ := correctness model p s hN injective
  refine ⟨final,run,?_,post.2⟩
  intro d
  rw [post.1 d,source_value p s q v input hN hS]

#axiomsClean update_eq
#axiomsClean final_eq
#axiomsClean source_value
#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.RealModel
