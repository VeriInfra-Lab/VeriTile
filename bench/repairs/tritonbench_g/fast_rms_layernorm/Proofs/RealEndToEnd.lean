import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealSequence

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealEndToEnd
open VeriTile.Triton
open RealGridForward (Address Heap read)
noncomputable section

def seed (saved upstream : Heap) : Heap := fun a => if a.1="dx" then upstream a else saved a
def weights (h : Call) (heap : Heap) (j : Fin h.cols) : ℝ := read (heap ("w",j.val))
def run (h : Call) (initial upstream : Heap) : Option (Heap × Heap) := do
  let forward ← RealGridForward.run h initial
  let backward ← RealSequence.run h (seed forward upstream)
  return (forward,backward)

theorem complete (h : Call) (initial upstream : Heap) :
    ∃ forward backward,run h initial upstream=some (forward,backward) ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (forward ("y",row.val*h.cols+j.val))=
        RealReference.output h.gemma h.eps (RealSequence.data h initial "x" row) (weights h initial) j) ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (backward ("dx",row.val*h.cols+j.val))=
        RealReference.inputGradient h.gemma h.eps (RealSequence.data h initial "x" row)
          (weights h initial) (RealSequence.data h upstream "dx" row) j) ∧
      (∀ j : Fin h.cols,read (backward ("dw",j.val))=
        RealReference.weightGradient h.eps (RealSequence.data h initial "x") (RealSequence.data h upstream "dx") j) ∧
      (∀ row : Fin h.rows,read (forward ("r",row.val))=RealReference.inv h.eps (RealSequence.data h initial "x" row)) ∧
      (∀ addr,(∀ row : Fin h.rows,¬RealGridBackward.writes h row addr) →
        (∀ pid : Fin (Geometry.featureCount h),¬RealGridWeight.writes h pid addr) →
        backward addr=seed forward upstream addr) := by
  obtain ⟨forward,hforward,fo,ff⟩:=RealGridForward.complete h initial
  obtain ⟨backward,hbackward,bo,bw,bf⟩:=RealSequence.complete h (seed forward upstream)
  have preserve (r : RegionName) (hr : r="x" ∨ r="w") (off : Nat) : forward (r,off)=initial (r,off) := by
    apply ff
    intro row
    rintro (⟨j,eq⟩|eq)
    · have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
    · have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
  have xs : RealSequence.data h (seed forward upstream) "x"=RealSequence.data h initial "x" := by
    funext row j
    simp only [RealSequence.data,seed,Prod.fst,if_neg (by decide : ¬(("x":RegionName)="dx"))]
    rw [preserve "x" (Or.inl rfl)]
  have ws : weights h (seed forward upstream)=weights h initial := by
    funext j
    simp only [weights,seed,Prod.fst,if_neg (by decide : ¬(("w":RegionName)="dx"))]
    rw [preserve "w" (Or.inr rfl)]
  have gs : RealSequence.data h (seed forward upstream) "dx"=RealSequence.data h upstream "dx" := by
    funext row j
    simp [RealSequence.data,seed]
  have saved (row : Fin h.rows) : read (seed forward upstream ("r",row.val))=
      RealReference.inv h.eps (RealSequence.data h (seed forward upstream) "x" row) := by
    rw [xs]
    simpa [seed,RealGridForward.address,RealGridForward.expected,RealRows.raw,RealGridForward.entry,RealSequence.data,
      BlockState.readMem,RealGridForward.read] using fo row (.inr ())
  refine ⟨forward,backward,by simp [run,hforward,hbackward],?_,?_,?_,?_,bf⟩
  · intro row j
    have result:=fo row (.inl j)
    have w : RealRows.raw h (RealGridForward.entry initial row.val h.rows) "w" 0=weights h initial := by
      funext k
      simp [RealRows.raw,weights,RealGridForward.entry,BlockState.readMem,RealGridForward.read]
      rfl
    change read (forward ("y",row.val*h.cols+j.val))=
      RealReference.output h.gemma h.eps _ _ j at result
    rw [w] at result
    exact result
  · intro row j
    have result:=bo row j
    unfold RealGridBackward.expected at result
    rw [saved row] at result
    have bridge:=RealReads.gradient_agrees h
      (RealGridBackward.raw h (seed forward upstream) row "x" h.cols)
      (RealGridBackward.raw h (seed forward upstream) row "w" 0)
      (RealGridBackward.raw h (seed forward upstream) row "dx" h.cols)
      (RealSequence.data h (seed forward upstream) "x" row) (weights h (seed forward upstream))
      (RealSequence.data h (seed forward upstream) "dx" row)
      (fun _ _ => rfl) (by intros; simp [RealGridBackward.raw,weights]) (fun _ _ => rfl) j
    rw [xs,ws,gs] at bridge
    rw [xs] at result
    exact result.trans bridge
  · intro j
    let pid:=Geometry.owner h j
    let lane : RealGridWeight.Output h pid:=⟨Geometry.lane j,by simpa [pid,Geometry.covered] using j.isLt⟩
    have result:=bw pid lane
    have formula:=RealSequence.weight_formula h (seed forward upstream) saved pid lane
    rw [xs,gs] at formula
    have indexEq : (⟨Geometry.column pid.val lane.val,lane.property⟩:Fin h.cols)=j := by
      apply Fin.ext
      exact Geometry.covered h j
    rw [indexEq] at formula
    have addrEq : RealGridWeight.address h pid lane=(("dw":RegionName),j.val) :=
      Prod.ext rfl (Geometry.covered h j)
    rw [addrEq] at result
    exact result.trans formula
  · intro row
    have result:=saved row
    rw [xs] at result
    simpa [seed] using result

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealEndToEnd
