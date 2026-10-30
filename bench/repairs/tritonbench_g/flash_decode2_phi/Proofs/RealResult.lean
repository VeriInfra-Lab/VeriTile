import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Coordinates

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Formula inputs are the supplied stage-1 summaries, indexed independently
of the source recurrence. No claim about their raw-token provenance is made. -/
def realLog (s : State (Option ℝ)) (g : Geometry) (row j : Nat) : ℝ :=
  ((s.mem "E" (row*g.capacity+j)).read .real).getD 0
def realMid (s : State (Option ℝ)) (g : Geometry) (row j d : Nat) : ℝ :=
  ((s.mem "M" (row*g.capacity*g.dimension+j*g.dimension+d)).read .real).getD 0

def FiniteInputs (s : State (Option ℝ)) (g : Geometry) : Prop :=
  (∀ row,row<g.rows → ∀ j,j<g.capacity → ∃ x,(s.mem "E" (row*g.capacity+j)).read .real=some x) ∧
  (∀ row,row<g.rows → ∀ j,j<g.capacity → ∀ d,d<g.dimension →
    ∃ x,(s.mem "M" (row*g.capacity*g.dimension+j*g.dimension+d)).read .real=some x)

def formula (s : State (Option ℝ)) (g : Geometry) (row d : Nat) : ℝ :=
  let K := count (Grid.setup g.batches g.heads row s) g.sequenceBlock
  Reference.target (realLog s g row) (fun j d => realMid s g row j d) K d

theorem ideal_output (q : Host.Request) (call : Host.Preparation (Option ℝ) q)
    (finite : FiniteInputs call.before q.g) (row : Nat) (hr : row<q.g.rows)
    (k : TileIndex [q.g.laneBlock]) (hk : lane k<q.g.dimension) :
    Grid.value Reference.ideal call.before q.g.masked q.g.batches q.g.heads q.g.capacity
      q.g.dimension q.g.sequenceBlock q.g.laneBlock row k=some (formula call.before q.g row (lane k)) := by
  let s := Grid.setup q.g.batches q.g.heads row call.before
  let x := realLog call.before q.g row
  let v := fun j (k : TileIndex [q.g.laneBlock]) =>
    if lane k<q.g.dimension then realMid call.before q.g row j (lane k) else 0
  have hc := Host.row_count q call row hr
  have hx : ∀ j,j<count s q.g.sequenceBlock → logvalue Reference.ideal s q.g.heads q.g.capacity j=some (x j) := by
    intro j hj
    have hjS : j<q.g.capacity := lt_of_lt_of_le hj hc.2
    obtain ⟨z,hz⟩ := finite.1 row hr j hjS
    simp only [logvalue,Reference.ideal,s,row_log_offset]
    change (call.before.mem "E" (row*q.g.capacity+j)).read .real=some (x j)
    simp [x,realLog,hz]
  have hv : ∀ j,j<count s q.g.sequenceBlock → ∀ k,
      value Reference.ideal s q.g.masked q.g.heads q.g.capacity q.g.dimension j k=some (v j k) := by
    intro j hj k
    have hm := Grid.active_iff q.g.masked q.g.dimension q.g.laneBlock q.valid.2.2.1 k
    by_cases hkd : lane k<q.g.dimension
    · have hjS : j<q.g.capacity := lt_of_lt_of_le hj hc.2
      obtain ⟨z,hz⟩ := finite.2 row hr j hjS (lane k) hkd
      simp only [value,Reference.ideal,hm,hkd,ite_true,s,row_mid_offset]
      change (call.before.mem "M" (row*q.g.capacity*q.g.dimension+j*q.g.dimension+lane k)).read .real=some (v j k)
      simp [v,realMid,hkd,hz]
    · simp [value,Reference.ideal,zero,hm,hkd,v]
  have h := Reference.source_correct s q.g.masked q.g.heads q.g.capacity q.g.dimension
    q.g.sequenceBlock q.g.laneBlock x v hc.1 hx hv k
  change output Reference.ideal s q.g.masked q.g.heads q.g.capacity q.g.dimension
    q.g.sequenceBlock q.g.laneBlock k=_
  rw [h]
  simp [formula,Reference.target,Reference.numerator,Reference.denominator,x,v,hk,s]

/-- Complete checked Python call: actual supplied output mutation, actual None
return, every output lane, all blocks from the real length buffer, and frame.
The numerical target is the independent unshifted weighted-summary formula. -/
theorem correct (q : Host.Request) (call : Host.Preparation (Option ℝ) q)
    (finite : FiniteInputs call.before q.g) (schedule : List Nat)
    (all : schedule.Perm (List.range q.g.rows)) :
    ∃ result,Host.invoke Reference.ideal q call schedule=some result ∧ result.returned=none ∧
      result.suppliedOutput=call.out ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.g.rows=0 then none else some call.owner) ∧
      (∀ row (k : TileIndex [q.g.laneBlock]),row<q.g.rows → lane k<q.g.dimension →
        result.state.mem "O" (row*q.g.dimension+lane k)=.mk .real
          (some (formula call.before q.g row (lane k)))) ∧
      (∀ r a,r≠("O" : RegionName) ∨ q.g.size≤a → result.state.mem r a=call.before.mem r a) := by
  obtain ⟨result,he,hn,ho,hctx,hl,hv,hf⟩ := Host.whole_typed Reference.ideal q call schedule all
  refine ⟨result,he,hn,ho,hctx,hl,?_,hf⟩
  intro row k hr hk
  rw [hv row k hr hk,ideal_output q call finite row hr k hk]

#axiomsClean ideal_output
#axiomsClean correct
#specNonCircular formula avoiding [Host.invoke,Grid.source,FP.Structural.exec,values]
#specNonCircular FiniteInputs avoiding [Host.invoke,Grid.source,FP.Structural.exec,values,formula]
end
end VeriTile.Bench.Repairs.FlashDecode2
