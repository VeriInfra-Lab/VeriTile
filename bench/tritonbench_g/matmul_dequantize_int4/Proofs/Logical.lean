import bench.tritonbench_g.matmul_dequantize_int4.Proofs.ISA
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
structure Legal (p : Parameters) : Prop where
  bk : 0<p.blockInner
  packed : p.blockInner/8*8=p.blockInner
  fullK : steps p*p.blockInner=p.inner
  fullGroup : p.groupSize/p.blockInner*p.blockInner=p.groupSize
  group : 0<p.groupSize
  branch : p.noGroups=decide (p.groupSize=p.inner)

def a (p : Parameters) (s : State Word) (r k : Nat) : Word :=
  (s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read .fp16
def q (p : Parameters) (s : State Word) (k n : Nat) : Int :=
  MatmulDequantInt4.FPTyped.nibble
    ((s.mem p.weight (k/8*p.weightInnerStride+n*p.weightColumnStride)).read .int) k
def z (p : Parameters) (s : State Word) (k n : Nat) : Int :=
  MatmulDequantInt4.FPTyped.nibble
    ((s.mem p.zero (k/p.groupSize*p.zeroGroupStride+n/8*p.zeroColumnStride)).read .int) n
def scale (p : Parameters) (s : State Word) (k n : Nat) : Word :=
  (s.mem p.scale (k/p.groupSize*p.scaleGroupStride+n*p.scaleColumnStride)).read .fp16
/-- Scalar half graph observed with default fusion: zero product rounds first,
then the weight product and subtraction form one half fused instruction. -/
def b (p : Parameters) (s : State Word) (k n : Nat) : Word :=
  QuantizeKvTransform.MixedBits.embed (FPCompiled.HalfFMA.fused (q p s k n) (z p s k n)
    (QuantizeKvTransform.MixedBits.half (scale p s k n)))
/-- Independent global-coordinate reference. It contains no pid, register,
source offset, execution, or final-memory definitions. -/
def accum (mma : ISA.MMA16) (p : Parameters) (s : State Word) (r n : Nat) : Nat → Word
  | 0 => FP.Value.zero .fp32
  | step+1 => ISA.chain mma
      (fun t : Fin p.blockInner => a p s r (step*p.blockInner+t.val))
      (fun t : Fin p.blockInner => b p s (step*p.blockInner+t.val) n)
      (accum mma p s r n step)
def output (mma : ISA.MMA16) (p : Parameters) (s : State Word) (r n : Nat) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (accum mma p s r n (steps p))

theorem k_bound (p : Parameters) (h : Legal p) (step : Nat) (hs : step<steps p)
    (t : Fin p.blockInner) : step*p.blockInner+t.val<p.inner := by
  have hh := Nat.mul_le_mul_right p.blockInner (show step+1≤steps p by omega)
  rw [Nat.add_mul,h.fullK] at hh
  have := t.isLt
  omega

theorem k_div (p : Parameters) (h : Legal p) (step : Nat) (t : Fin p.blockInner) :
    (step*p.blockInner+t.val)/p.blockInner=step := by
  simpa [Nat.add_comm,Nat.mul_comm,Nat.div_eq_of_lt t.isLt] using
    Nat.add_mul_div_left t.val step h.bk

theorem group_eq (p : Parameters) (h : Legal p) (step : Nat) (hs : step<steps p)
    (t : Fin p.blockInner) : groupIndex p step=(step*p.blockInner+t.val)/p.groupSize := by
  by_cases he : p.groupSize=p.inner
  · simp only [groupIndex,h.branch,he,decide_true,↓reduceIte]
    exact (Nat.div_eq_of_lt (k_bound p h step hs t)).symm
  · simp only [groupIndex,h.branch,he,decide_false,↓reduceIte]
    calc
      step/(p.groupSize/p.blockInner) =
          ((step*p.blockInner+t.val)/p.blockInner)/(p.groupSize/p.blockInner) := by rw [k_div p h]
      _ = (step*p.blockInner+t.val)/(p.blockInner*(p.groupSize/p.blockInner)) := Nat.div_div_eq_div_mul _ _ _
      _ = _ := by rw [Nat.mul_comm p.blockInner,h.fullGroup]

theorem packed_div (p : Parameters) (h : Legal p) (step t : Nat) :
    (step*p.blockInner+t)/8=step*(p.blockInner/8)+t/8 := by
  conv_lhs => rw [←h.packed]
  rw [←Nat.mul_assoc,Nat.add_comm,Nat.add_mul_div_right _ _ (by decide : 0<8)]
  omega

theorem packed_mod (p : Parameters) (h : Legal p) (step t : Nat) :
    (step*p.blockInner+t)%8=t%8 := by
  conv_lhs => rw [←h.packed]
  simp [Nat.add_mod,Nat.mul_mod]

theorem a_global (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p)
    (hr : inputRow p s i.1<p.rows) :
    FPSoftware.machineA p s step i=a p s (inputRow p s i.1) (step*p.blockInner+i.2.1.val) := by
  simp only [FPSoftware.machineA,hr,↓reduceIte,a,aOffset,inputInner]
  congr 2
  ring

theorem b_global (p : Parameters) (h : Legal p) (s : State Word) (step : Nat)
    (hs : step<steps p) (i : BIndex p) :
    FPFused.fusedWeight p s step i=b p s (step*p.blockInner+i.1.val) (inputColumn p s i.2.1) := by
  have hg := group_eq p h step hs i.1
  have hq := packed_div p h step i.1.val
  have hm := packed_mod p h step i.1.val
  have ho : bOffset p s step i=(step*p.blockInner+i.1.val)/8*p.weightInnerStride+
      inputColumn p s i.2.1*p.weightColumnStride := by
    rw [hq]
    simp only [bOffset,inputInner]
    ring
  simp only [FPFused.fusedWeight,b,q,z,scale,weightNibble,zeroNibble,scaleValue,
    scaleOffset,zeroOffset,ho,hg]
  simp only [MatmulDequantInt4.FPTyped.nibble,hm]
  simp only [Nat.add_comm]

theorem accum_global (mma : ISA.MMA16) (p : Parameters) (h : Legal p) (s : State Word)
    (i : CIndex p) (hr : inputRow p s i.1<p.rows) (count : Nat) (hc : count≤steps p) :
    ISA.accum mma p s count i=accum mma p s (inputRow p s i.1) (inputColumn p s i.2.1) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [ISA.accum,accum,ih (by omega)]
    congr 1
    · funext t
      exact a_global p s count (i.1,t,PUnit.unit) hr
    · funext t
      exact b_global p h s count (by omega) (t,i.2.1,PUnit.unit)

theorem output_global (mma : ISA.MMA16) (p : Parameters) (h : Legal p) (s : State Word)
    (i : CIndex p) (hi : active p s i) :
    ISA.output mma p s i=output mma p s (inputRow p s i.1) (inputColumn p s i.2.1) := by
  unfold ISA.output output
  rw [accum_global mma p h s i hi.1 _ (by omega)]

theorem output_preserved (mma : ISA.MMA16) (p : Parameters) (s t : State Word)
    (ha : p.input≠p.output) (hq : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (h : Outside p.output s t) : output mma p t=output mma p s := by
  have ea : ∀ r k,a p t r k=a p s r k := by intro r k; simp [a,h _ _ ha]
  have eb : ∀ k n,b p t k n=b p s k n := by
    intro k n
    simp only [b,q,z,scale,h _ _ hq,h _ _ hs,h _ _ hz]
  have eacc (r n count : Nat) : accum mma p t r n count=accum mma p s r n count := by
    induction count with
    | zero => rfl
    | succ k ih => simp only [accum,ea,eb,ih]
  funext r n
  simp only [output,eacc]

#specNonCircular output avoiding [ISA.output,ISA.accum,FPFused.exec,aOffset,bOffset,groupIndex]
#axiomsClean group_eq
#axiomsClean packed_div
#axiomsClean b_global
#axiomsClean output_global
#axiomsClean output_preserved
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Logical
