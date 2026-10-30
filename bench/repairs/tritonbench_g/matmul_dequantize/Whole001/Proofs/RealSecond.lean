import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealBase

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealSecond
open VeriTile.Triton FP.Structural
open FPTypedSecond
open Real
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Read logical tensor coordinates through the caller's actual strides. -/
def inputs (p : Parameters) (s : State ℝ) : RawInputs where
  a := fun m k => (s.mem p.input (m*p.inputRowStride+k*p.inputInnerStride)).read .fp32
  weight := fun k n => (s.mem p.weight (k*p.weightInnerStride+n*p.weightColumnStride)).read .int
  zero := fun g n => (s.mem p.zero (g*p.zeroGroupStride+n*p.zeroColumnStride)).read .int
  scale := fun g n => (s.mem p.scale (g*p.scaleGroupStride+n*p.scaleColumnStride)).read .fp32

def outputRow (p : Parameters) (s : State ℝ) (i : CIndex p) := rowBlock p s*p.blockRows+i.1.val
def outputColumn (p : Parameters) (s : State ℝ) (i : CIndex p) := columnBlock p s*p.blockColumns+i.2.1.val

theorem lane_product (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32) (hpid : s.pids 1=0)
    (i : CIndex p) (hi : active p s i) (step : Nat) (e : Fin p.blockInner) :
    aValue scalar p s step (i.1,e,())*bValue scalar p s step (e,i.2.1,())=
      if step*32+e.val<p.inner then
        (inputs p s).a (outputRow p s i) (step*32+e.val)*
          rawDequant (inputs p s) p.groupSize (step*32+e.val) (outputColumn p s i) else 0 := by
  have hr : inputRow p s i.1=outputRow p s i := Nat.mod_eq_of_lt hi.1
  have hc : inputColumn p s i.2.1=outputColumn p s i := Nat.mod_eq_of_lt hi.2
  have he : inputInner p s e=e.val := by simp [inputInner,hpid]
  have hm : (step*32+e.val)%8=e.val%8 := by omega
  have hq : (step*32+e.val)/8=step*4+e.val/8 := by omega
  have hstride : 32*p.weightInnerStride/8=4*p.weightInnerStride := by omega
  have haOff : aOffset p s step (i.1,e,())=
      outputRow p s i*p.inputRowStride+(step*32+e.val)*p.inputInnerStride := by
    simp only [aOffset,he,hr,hBK]
    ring
  have hbOff : bOffset p s step (e,i.2.1,())=
      ((step*32+e.val)/8)*p.weightInnerStride+outputColumn p s i*p.weightColumnStride := by
    simp only [bOffset,he,hc,hBK,hstride,hq]
    ring
  have hsOff : scaleOffset p s step (e,i.2.1,())=
      ((step*32+e.val)/p.groupSize)*p.scaleGroupStride+outputColumn p s i*p.scaleColumnStride := by
    simp only [scaleOffset,he,hc,hBK,Nat.add_comm]
  have hzOff : zeroOffset p s step (e,i.2.1,())=
      ((step*32+e.val)/p.groupSize)*p.zeroGroupStride+outputColumn p s i/8*p.zeroColumnStride := by
    simp only [zeroOffset,he,hc,hBK,Nat.add_comm]
  have hactive : innerActive p s step e ↔ step*32+e.val<p.inner := by
    simp only [innerActive,he,hBK,Nat.add_comm]
  by_cases hk : step*32+e.val<p.inner
  · simp [aValue,bValue,scalar,difference,packedLoad,hactive,hk,haOff,hbOff,hsOff,hzOff,
      rawDequant,inputs,he,hc,signedNibble,
      VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble,hm]
  · simp [aValue,bValue,scalar,difference,packedLoad,hactive,hk]
    left
    norm_num

theorem block_dot (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32) (hpid : s.pids 1=0)
    (i : CIndex p) (hi : active p s i) (step : Nat) :
    secondDot.apply (some .fp32) {} (batch:=[]) (aValue scalar p s step)
      (fun j => scalar.cast (some .fp32) .fp16 .fp16 (bValue scalar p s step j)) i=
      sum (fun e => if step*32+e<p.inner then
        (inputs p s).a (outputRow p s i) (step*32+e)*
          rawDequant (inputs p s) p.groupSize (step*32+e) (outputColumn p s i) else 0) 32 := by
  change (List.finRange p.blockInner).foldl
    (fun t e => t+aValue scalar p s step (i.1,e,())*bValue scalar p s step (e,i.2.1,())) 0=_
  rw [dot_sum p.blockInner _ _
    (fun e => if step*32+e<p.inner then
      (inputs p s).a (outputRow p s i) (step*32+e)*
        rawDequant (inputs p s) p.groupSize (step*32+e) (outputColumn p s i) else 0)
    (lane_product p s hBK hpid i hi step),hBK]

theorem accumulation_formula (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32) (hpid : s.pids 1=0)
    (i : CIndex p) (hi : active p s i) (steps : Nat) :
    accumulation scalar secondDot p s steps i=
      chunks (fun k => if k<p.inner then (inputs p s).a (outputRow p s i) k*
        rawDequant (inputs p s) p.groupSize k (outputColumn p s i) else 0) 32 steps := by
  induction steps with
  | zero => rfl
  | succ n ih =>
    change accumulation scalar secondDot p s n i+
      secondDot.apply (some .fp32) {} (batch:=[]) (aValue scalar p s n)
        (fun j => scalar.cast (some .fp32) .fp16 .fp16 (bValue scalar p s n j)) i=_
    rw [ih,block_dot p s hBK hpid i hi n]
    rfl

theorem output_formula (p : Parameters) (s : State ℝ) (hBK : p.blockInner=32) (hpid : s.pids 1=0)
    (hsteps : p.steps=(p.inner+31)/32) (i : CIndex p) (hi : active p s i) :
    outputValue scalar secondDot p s i=
      rawMatmul (inputs p s) p.inner p.groupSize (outputRow p s i) (outputColumn p s i) := by
  change accumulation scalar secondDot p s p.steps i=_
  rw [accumulation_formula p s hBK hpid i hi,chunks_eq,sum_padding]
  have hcover : p.inner≤p.steps*32 := by
    rw [hsteps]
    have h := VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.LogicalSum.ceil_cover p.inner 32 (by decide)
    simpa using h
  rw [Nat.min_eq_left hcover]
  rfl

#axiomsClean lane_product
#axiomsClean output_formula
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealSecond
