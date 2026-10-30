import bench.tritonbench_g.matmul_dequantize.Proofs.RealBase

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealFirst
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst
open Real
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def inputs (p : Parameters) (s : State ℝ) : RawInputs where
  a := fun m k => (s.mem p.input (m*p.inputRowStride+k*p.inputInnerStride)).read .fp16
  weight := fun k n => (s.mem p.weight (k*p.weightInnerStride+n*p.weightColumnStride)).read .int
  zero := fun g n => (s.mem p.zero (g*p.zeroGroupStride+n*p.zeroColumnStride)).read .int
  scale := fun g n => (s.mem p.scale (g*p.scaleGroupStride+n*p.scaleColumnStride)).read .fp16

def outputRow (p : Parameters) (s : State ℝ) (i : CIndex p) := rowBlock p s*p.blockRows+i.1.val
def outputColumn (p : Parameters) (s : State ℝ) (i : CIndex p) := columnBlock p s*p.blockColumns+i.2.1.val

theorem group_exact (p : Parameters) (hBK : p.blockInner=32) (G : Nat) (hG : p.groupSize=32*G)
    (hmode : p.noGroups=Bool.true → p.groupSize=p.inner) (hK : p.inner=steps p*32)
    (step : Nat) (hs : step<steps p) (e : Fin p.blockInner) :
    groupIndex p step=(step*32+e.val)/p.groupSize := by
  have he : e.val<32 := by simpa [hBK] using e.isLt
  have hk : step*32+e.val<p.inner := by omega
  by_cases hn : p.noGroups=Bool.true
  · simp only [groupIndex,hn,ite_true,hmode hn,Nat.div_eq_of_lt hk]
  · have hq : (step*32+e.val)/32=step := by omega
    have hg : p.groupSize/32=G := by omega
    simp only [groupIndex,hn,ite_false,hBK,hg,hG]
    rw [← Nat.div_div_eq_div_mul,hq]
    simp

theorem lane_product (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32)
    (G : Nat) (hG : p.groupSize=32*G) (hmode : p.noGroups=Bool.true → p.groupSize=p.inner)
    (hK : p.inner=steps p*32) (i : CIndex p) (hi : active p s i)
    (step : Nat) (hs : step<steps p) (e : Fin p.blockInner) :
    aValue scalar p s step (i.1,e,())*bValue scalar p s step (e,i.2.1,())=
      (inputs p s).a (outputRow p s i) (step*32+e.val)*
        rawDequant (inputs p s) p.groupSize (step*32+e.val) (outputColumn p s i) := by
  have hr : inputRow p s i.1=outputRow p s i := rfl
  have hc : inputColumn p s i.2.1=outputColumn p s i := rfl
  have he : inputInner p s e=e.val := rfl
  have hm : (step*32+e.val)%8=e.val%8 := by omega
  have hq : (step*32+e.val)/8=step*4+e.val/8 := by omega
  have hg := group_exact p hBK G hG hmode hK step hs e
  have haOff : aOffset p s step (i.1,e,())=
      outputRow p s i*p.inputRowStride+(step*32+e.val)*p.inputInnerStride := by
    simp only [aOffset,he,hr,hBK]
    ring
  have hbOff : bOffset p s step (e,i.2.1,())=
      ((step*32+e.val)/8)*p.weightInnerStride+outputColumn p s i*p.weightColumnStride := by
    simp only [bOffset,he,hc,hBK,hq]
    norm_num
    ring
  have hsOff : scaleOffset p s step (i.2.1,())=
      ((step*32+e.val)/p.groupSize)*p.scaleGroupStride+outputColumn p s i*p.scaleColumnStride := by
    simp only [scaleOffset,hc,hg,Nat.add_comm]
  have hzOff : zeroOffset p s step (i.2.1,())=
      ((step*32+e.val)/p.groupSize)*p.zeroGroupStride+outputColumn p s i/8*p.zeroColumnStride := by
    simp only [zeroOffset,hc,hg,Nat.add_comm]
  have hrow : inputRow p s i.1<p.rows := hi.1
  simp only [aValue,bValue,scalar,weightNibble,zeroNibble,scaleValue,zeroValue,hrow,ite_true,
    haOff,hbOff,hsOff,hzOff,rawDequant,inputs,he,hc,signedNibble,
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble,hm,Int.cast_sub]
  ring

theorem block_dot (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32)
    (G : Nat) (hG : p.groupSize=32*G) (hmode : p.noGroups=Bool.true → p.groupSize=p.inner)
    (hK : p.inner=steps p*32) (i : CIndex p) (hi : active p s i) (step : Nat) (hs : step<steps p) :
    firstDot.apply (some .fp32) {} (batch:=[]) (aValue scalar p s step) (bValue scalar p s step) i=
      sum (fun e => (inputs p s).a (outputRow p s i) (step*32+e)*
        rawDequant (inputs p s) p.groupSize (step*32+e) (outputColumn p s i)) 32 := by
  change (List.finRange p.blockInner).foldl
    (fun t e => t+aValue scalar p s step (i.1,e,())*bValue scalar p s step (e,i.2.1,())) 0=_
  rw [dot_sum p.blockInner _ _
    (fun e => (inputs p s).a (outputRow p s i) (step*32+e)*
      rawDequant (inputs p s) p.groupSize (step*32+e) (outputColumn p s i))
    (lane_product p s hBK G hG hmode hK i hi step hs),hBK]

theorem accumulation_formula (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32)
    (G : Nat) (hG : p.groupSize=32*G) (hmode : p.noGroups=Bool.true → p.groupSize=p.inner)
    (hK : p.inner=steps p*32) (i : CIndex p) (hi : active p s i) (n : Nat) (hn : n≤steps p) :
    accumulation scalar firstDot p s n i=
      chunks (fun k => (inputs p s).a (outputRow p s i) k*
        rawDequant (inputs p s) p.groupSize k (outputColumn p s i)) 32 n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change accumulation scalar firstDot p s n i+
      firstDot.apply (some .fp32) {} (batch:=[]) (aValue scalar p s n) (bValue scalar p s n) i=_
    rw [ih (by omega),block_dot p s hBK G hG hmode hK i hi n (by omega)]
    rfl

theorem output_formula (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32)
    (G : Nat) (hG : p.groupSize=32*G) (hmode : p.noGroups=Bool.true → p.groupSize=p.inner)
    (hK : p.inner=steps p*32) (i : CIndex p) (hi : active p s i) :
    outputValue scalar firstDot p s i=
      rawMatmul (inputs p s) p.inner p.groupSize (outputRow p s i) (outputColumn p s i) := by
  change accumulation scalar firstDot p s (steps p) i=_
  rw [accumulation_formula p s hBK G hG hmode hK i hi (steps p) (by omega),chunks_eq,← hK]
  rfl

#axiomsClean group_exact
#axiomsClean lane_product
#axiomsClean output_formula
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealFirst
