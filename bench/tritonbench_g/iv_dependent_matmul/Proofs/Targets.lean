import bench.tritonbench_g.iv_dependent_matmul.Proofs.Instructions

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Target
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
open TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000

/-- One fixed emitted MMA result lane on its raw FP32 operand words. The
opcode itself specifies TF32 operand interpretation. Saved SM80 PTX has no
separate cvt.tf32 instruction, so this model inserts no such conversion. -/
structure NativeProfile where
  mmaM16N8K8TF32 : Instructions.MMA8

def NativeProfile.reference (p : NativeProfile) : Instructions.Reference :=
  ⟨.ieeeValue, id, p.mmaM16N8K8TF32⟩

def mathematical (K : Nat) (A B : Nat → Nat → ℝ) (i j : Nat) : ℝ :=
  ∑ k : Fin K, A i k.val * B k.val j

/-- Operand functions, selected K8 instruction atoms, zero-tail lanes, a live
FP32 seed, and the terminal half/single conversions define the native target.
There is no source state, output buffer or interpreter invocation here. -/
def block (I : Instructions.Native) (K BK : Nat) (A B : Nat → Nat → Word)
    (i j t : Nat) (seed : Word) : Word :=
  Instructions.dot I BK
    (fun k => if t*BK+k.val<K then A i (t*BK+k.val) else FP.Value.zero .fp32)
    (fun k => if t*BK+k.val<K then B (t*BK+k.val) j else FP.Value.zero .fp32) seed

def native (R : Instructions.Reference) (K BK : Nat) (A B : Nat → Nat → Word)
    (i j : Nat) : Word :=
  let I := Instructions.referenceInstructions R
  I.widen (I.narrow (foldRange (block I K BK A B i j) 0 ((K+BK-1)/BK) 1 (FP.Value.zero .fp32)))

def inputA (r : Request) (s : State Word) (i k : Nat) : Word :=
  (s.mem r.A (i*max r.inner 1+k)).read .fp32
def inputB (r : Request) (s : State Word) (k j : Nat) : Word :=
  (s.mem r.B (k*max r.columns 1+j)).read .fp32

theorem operands (r : Request) (s : State Word) (pid t : Nat) (q : CIndex r)
    (hr : row r.parameters (instanceState pid s) q.1<r.rows)
    (hc : col r.parameters (instanceState pid s) q.2.1<r.columns) :
    (fun k => aValues r.parameters (instanceState pid s) t (q.1,k,PUnit.unit)) =
      (fun k : Fin r.blockInner => if t*r.blockInner+k.val<r.inner then
        inputA r s (row r.parameters (instanceState pid s) q.1) (t*r.blockInner+k.val)
        else FP.Value.zero .fp32) ∧
    (fun k => bValues r.parameters (instanceState pid s) t (k,q.2.1,PUnit.unit)) =
      (fun k : Fin r.blockInner => if t*r.blockInner+k.val<r.inner then
        inputB r s (t*r.blockInner+k.val) (col r.parameters (instanceState pid s) q.2.1)
        else FP.Value.zero .fp32) := by
  have ha (k : Fin r.blockInner) :
      aOffset r.parameters (instanceState pid s) t (q.1,k,PUnit.unit)=
        row r.parameters (instanceState pid s) q.1*max r.inner 1+(t*r.blockInner+k.val) := by
    unfold aOffset aBase
    change (row r.parameters (instanceState pid s) q.1%r.rows)*max r.inner 1+k.val*1+t*r.blockInner*1=_
    rw [Nat.mod_eq_of_lt hr]
    ring
  have hb (k : Fin r.blockInner) :
      bOffset r.parameters (instanceState pid s) t (k,q.2.1,PUnit.unit)=
        (t*r.blockInner+k.val)*max r.columns 1+col r.parameters (instanceState pid s) q.2.1 := by
    unfold bOffset bBase
    change k.val*max r.columns 1+(col r.parameters (instanceState pid s) q.2.1%r.columns)*1+t*r.blockInner*max r.columns 1=_
    rw [Nat.mod_eq_of_lt hc]
    ring
  constructor
  · funext k
    have guard : k.val<r.inner-t*r.blockInner ↔ t*r.blockInner+k.val<r.inner := by omega
    simp only [aValues]
    rw [ha k]
    simp only [Request.parameters,guard,inputA,instanceState]
  · funext k
    have guard : k.val<r.inner-t*r.blockInner ↔ t*r.blockInner+k.val<r.inner := by omega
    simp only [bValues]
    rw [hb k]
    simp only [Request.parameters,guard,inputB,instanceState]

theorem update_at (R : Instructions.Reference) (r : Request) (s : State Word)
    (pid t : Nat) (q : CIndex r)
    (hr : row r.parameters (instanceState pid s) q.1<r.rows)
    (hc : col r.parameters (instanceState pid s) q.2.1<r.columns) (acc : Accumulator r.parameters) :
    update R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
      (Instructions.accumulatorPort (Instructions.referenceInstructions R)) .accumulator
      r.parameters (instanceState pid s) t acc q =
    block (Instructions.referenceInstructions R) r.inner r.blockInner (inputA r s) (inputB r s)
      (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1) t (acc q) := by
  simp only [update, Instructions.accumulatorPort, Instructions.dotValue]
  rw [(operands r s pid t q hr hc).1,(operands r s pid t q hr hc).2]
  rfl

theorem range_at (R : Instructions.Reference) (r : Request) (s : State Word)
    (pid cur : Nat) (q : CIndex r)
    (hr : row r.parameters (instanceState pid s) q.1<r.rows)
    (hc : col r.parameters (instanceState pid s) q.2.1<r.columns) (acc : Accumulator r.parameters) :
    foldRange (update R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
      (Instructions.accumulatorPort (Instructions.referenceInstructions R)) .accumulator
      r.parameters (instanceState pid s)) cur (count r.parameters) 1 acc q =
    foldRange (block (Instructions.referenceInstructions R) r.inner r.blockInner (inputA r s) (inputB r s)
      (row r.parameters (instanceState pid s) q.1) (col r.parameters (instanceState pid s) q.2.1))
      cur (count r.parameters) 1 (acc q) := by
  by_cases h : cur<count r.parameters
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,h,↓reduceDIte]
    rw [range_at R r s pid (cur+1) q hr hc,update_at R r s pid cur q hr hc]
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,h,↓reduceDIte]
termination_by count r.parameters-cur
decreasing_by omega

theorem reference_eq (R : Instructions.Reference) (r : Request) (s : State Word)
    (i j : Nat) (hi : i<r.rows) (hj : j<r.columns) :
    reference R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
      (Instructions.accumulatorPort (Instructions.referenceInstructions R)) .accumulator r s i j =
    native R r.inner r.blockInner (inputA r s) (inputB r s) i j := by
  obtain ⟨pid,_,q,hr,hc⟩ := cell_covered r s i j hi hj
  have canonical := canonical_of_lane r s pid q
  rw [hr,hc] at canonical
  unfold reference
  rw [canonical.1,canonical.2]
  unfold outputValues accumulated
  rw [range_at R r s pid 0 q (by simpa [hr] using hi) (by simpa [hc] using hj),hr,hc]
  rfl

#specNonCircular mathematical avoiding [FPValue.accumulated, FP.Structural.exec]
#specNonCircular native avoiding [FPValue.accumulated, Whole.reference, FP.Structural.exec]
#axiomsClean operands
#axiomsClean update_at
#axiomsClean range_at
#axiomsClean reference_eq
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Target
