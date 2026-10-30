import bench.tritonbench_g.matmul_dequantize.Proofs.SecondPublicResult
import bench.tritonbench_g.matmul_dequantize.Proofs.RealSecond

namespace VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open FPTypedSecond (Parameters)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem mathematical_binding (p : Parameters) (jobs : List Nat) (vals : Inputs ℝ p) (s : State ℝ)
    (bound : (realIO p jobs).InputBinding vals s)
    (hBK : p.blockInner=32) (hsteps : p.steps=(p.inner+31)/32)
    (j : OutputLane p) (hj : Geometry.Active (SecondGrid.geometry p) j.1.val j.2) :
    FPTypedSecond.outputValue Real.scalar Real.secondDot p (GridExecution.atProgram s j.1.val) j.2=
      mathematical p vals ⟨0,by simp [outputChannels]⟩ j := by
  rw [RealSecond.output_formula p (GridExecution.atProgram s j.1.val) hBK (by simp [GridExecution.atProgram]) hsteps j.2 hj]
  change Real.rawMatmul (RealSecond.inputs p (GridExecution.atProgram s j.1.val)) p.inner p.groupSize
    (row p j.1.val j.2.1.val) (column p j.1.val j.2.2.1.val)=
    Real.rawMatmul (rawInputs p vals) p.inner p.groupSize
      (row p j.1.val j.2.1.val) (column p j.1.val j.2.2.1.val)
  apply Real.sum_congr
  intro k hk
  have he : k%32<p.blockInner := by omega
  let e : Fin p.blockInner := ⟨k%32,he⟩
  have hs : k/32<p.steps := by rw [hsteps]; omega
  have hm : row p j.1.val j.2.1.val<p.rows := hj.1
  have hn : column p j.1.val j.2.2.1.val<p.columns := hj.2
  have decomp : e.val+k/32*32=k := by dsimp [e]; omega
  have active : innerActive p (k/32) e.val := by simpa only [innerActive,hBK,decomp] using hk
  have ha := a_read Real.scalar Real.secondDot p jobs vals s bound 0 j.1.val j.1.isLt
    (k/32) hs (j.2.1,e,()) active
  have hb := b_read Real.scalar Real.secondDot p jobs vals s bound 0 j.1.val j.1.isLt
    (k/32) hs (e,j.2.2.1,()) active
  have hc := scale_read Real.scalar Real.secondDot p jobs vals s bound 0 j.1.val j.1.isLt
    (k/32) hs (e,j.2.2.1,()) active
  have hz := zero_read Real.scalar Real.secondDot p jobs vals s bound 0 j.1.val j.1.isLt
    (k/32) hs (e,j.2.2.1,()) active
  have ae : aAddress p j.1.val (k/32) j.2.1.val e.val=
      row p j.1.val j.2.1.val*p.inputRowStride+k*p.inputInnerStride := by
    simp only [aAddress,hBK,Nat.mod_eq_of_lt hm]
    rw [Nat.add_assoc,show e.val*p.inputInnerStride+k/32*(32*p.inputInnerStride)=k*p.inputInnerStride by
      rw [←Nat.mul_assoc,←Nat.add_mul,decomp]]
  have be : bAddress p j.1.val (k/32) e.val j.2.2.1.val=
      k/8*p.weightInnerStride+column p j.1.val j.2.2.1.val*p.weightColumnStride := by
    have heq : e.val/8+k/32*4=k/8 := by dsimp [e]; omega
    have quotient : 32*p.weightInnerStride/8=4*p.weightInnerStride := by omega
    simp only [bAddress,hBK,Nat.mod_eq_of_lt hn,quotient]
    calc
      _ = (e.val/8+k/32*4)*p.weightInnerStride+column p j.1.val j.2.2.1.val*p.weightColumnStride := by ring
      _ = _ := by rw [heq]
  have se : sAddress p j.1.val (k/32) e.val j.2.2.1.val=
      k/p.groupSize*p.scaleGroupStride+column p j.1.val j.2.2.1.val*p.scaleColumnStride := by
    simp [sAddress,hBK,decomp,Nat.mod_eq_of_lt hn]
  have ze : zAddress p j.1.val (k/32) e.val j.2.2.1.val=
      k/p.groupSize*p.zeroGroupStride+column p j.1.val j.2.2.1.val/8*p.zeroColumnStride := by
    simp [zAddress,hBK,decomp,Nat.mod_eq_of_lt hn]
  simp only [ae] at ha
  simp only [be] at hb
  simp only [se] at hc
  simp only [ze] at hz
  simp only [Real.rawDequant,RealSecond.inputs,rawInputs,GridExecution.atProgram]
  rw [ha,hb,hc,hz]

/-- Same independent matrix formula and original Real conditions, now on
finite typed operands with the actual complete safe source schedule. -/
theorem real_implements (p : Parameters) (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (hBK : p.blockInner=32) (hsteps : p.steps=(p.inner+31)/32)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    (realIO p jobs).Implements (mathematical p) := by
  intro vals _ bounds backed s bound
  obtain ⟨safe,t,ran,values,frame⟩ := expression_implements Real.scalar Real.secondDot p valid ha hb hs hz jobs schedule
    vals trivial bounds backed s bound
  refine ⟨safe,t,ran,?_,frame⟩
  intro o j hj
  fin_cases o
  have equal := (expression_binding Real.scalar Real.secondDot p jobs vals s bound j).symm.trans
    (mathematical_binding p jobs vals s bound hBK hsteps j hj)
  change mapCell id (t.mem p.output (Geometry.offset (SecondGrid.geometry p) j.1.val j.2))=
    Cell.mk .fp16 (mathematical p vals ⟨0,by simp [outputChannels]⟩ j)
  rw [←equal]
  exact values ⟨0,by simp [program,memory,outputChannels]⟩ j hj

#axiomsClean mathematical_binding
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
