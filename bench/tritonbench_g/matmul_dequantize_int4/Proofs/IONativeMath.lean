import bench.tritonbench_g.matmul_dequantize_int4.Proofs.IOData

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeMath
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section

/-- Independent coordinate dequantization, with the retained rounded zero
product, half sign-bit negation and one half FMA. -/
def dequant (c : RawKernel.Layout) (x : IOData.Arrays Word) (k n : Nat) : Word :=
  QuantizeKvTransform.MixedBits.embed (MatmulDequantizeInt4.FPCompiled.HalfFMA.fused
    (MatmulDequantInt4.FPTyped.nibble (x.Q (k/8) n) k)
    (MatmulDequantInt4.FPTyped.nibble (x.Z (k/c.group) (n/8)) n)
    (QuantizeKvTransform.MixedBits.half (x.S (k/c.group) n)))

/-- Fixed K16 instruction lanes and live FP32 carry in the selected tuning's
source order. This formula contains no source states, addresses or execution. -/
def accumulation (mma : ISA.MMA16) (c : RawKernel.Layout) (i : Fin 17)
    (x : IOData.Arrays Word) (r n : Nat) : Nat → Word
  | 0=>FP.Value.zero .fp32
  | step+1=>ISA.chain mma
      (fun k : Fin (tunings i).BK=>x.A r (step*(tunings i).BK+k.val))
      (fun k : Fin (tunings i).BK=>dequant c x (step*(tunings i).BK+k.val) n)
      (accumulation mma c i x r n step)

def value (mma : ISA.MMA16) (c : RawKernel.Layout) (i : Fin 17)
    (x : IOData.Arrays Word) (r n : Nat) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (accumulation mma c i x r n ((c.K+(tunings i).BK-1)/(tunings i).BK))

def formula (mma : Opcode → ISA.MMA16) (c : RawKernel.Layout) (i : Fin 17)
    (x : ChannelValues Word (IOData.inputs c)) :=
  IOData.expected c (value (mma (tunings i).opcode) c i (IOData.arrays c x))

theorem b_bound (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation Word)) (x : ChannelValues Word (IOData.inputs c)) (s : State Word)
    (binding : (IOData.program Word c jobs).InputBinding x s)
    (k n : Nat) (hk : k<c.K) (hn : n<c.N) :
    Logical.b (params c.call i) s k n=dequant c (IOData.arrays c x) k n := by
  have hg := Bounds.group_bound c.call (c.valid valid) k hk
  change k/c.group<c.K/c.group at hg
  have hk8 : k/8<c.K/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣128) valid.k128)] using hk
  have hn8 : n/8<c.N/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣256) valid.n256)] using hn
  change QuantizeKvTransform.MixedBits.embed (MatmulDequantizeInt4.FPCompiled.HalfFMA.fused
    (MatmulDequantInt4.FPTyped.nibble ((s.mem "Q" (c.Q.addr (k/8) n)).read .int) k)
    (MatmulDequantInt4.FPTyped.nibble ((s.mem "Z" (c.Z.addr (k/c.group) (n/8))).read .int) n)
    (QuantizeKvTransform.MixedBits.half ((s.mem "S" (c.S.addr (k/c.group) n)).read .fp16)))=_
  rw [IOData.binding_q c jobs x s binding _ _ hk8 hn,
    IOData.binding_z c jobs x s binding _ _ hg hn8,
    IOData.binding_s c jobs x s binding _ _ hg hn]
  rfl

theorem output (mma : ISA.MMA16) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation Word)) (x : ChannelValues Word (IOData.inputs c)) (s : State Word)
    (binding : (IOData.program Word c jobs).InputBinding x s)
    (r n : Nat) (hr : r<c.M) (hn : n<c.N) :
    Logical.output mma (params c.call i) s r n=value mma c i (IOData.arrays c x) r n := by
  have acc (count : Nat) (hc : count≤steps (params c.call i)) :
      Logical.accum mma (params c.call i) s r n count=
        accumulation mma c i (IOData.arrays c x) r n count := by
    induction count with
    | zero=>rfl
    | succ count ih=>
      simp only [Logical.accum,accumulation,ih (by omega)]
      congr 1
      · funext k
        exact IOData.binding_a c jobs x s binding _ _ hr
          (Logical.k_bound _ (legal c.call (c.valid valid) i) count (by omega) k)
      · funext k
        exact b_bound c valid i jobs x s binding _ _
          (Logical.k_bound _ (legal c.call (c.valid valid) i) count (by omega) k) hn
  unfold Logical.output value
  rw [acc _ (by omega)]
  rfl

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeMath
