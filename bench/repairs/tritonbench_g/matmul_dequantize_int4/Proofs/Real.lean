import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Host
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Packing
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.RealSum
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Real
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open MatmulKernel.RealSum
noncomputable section
set_option maxHeartbeats 4000000
/-- Mathematical input coordinates; no source pid, register, offsets or output. -/
structure Input where
  A : Nat → Nat → ℝ
  Q : Nat → Nat → Packing.Word
  S : Nat → Nat → ℝ
  Z : Nat → Nat → Packing.Word

def nibble (w : Packing.Word) (k : Nat) : ℝ :=
  (Packing.nibble w ⟨k%8,Nat.mod_lt _ (by decide)⟩).toNat
/-- Independent GPTQ dequantization and matrix product. -/
def dequant (c : Call) (x : Input) (k n : Nat) : ℝ :=
  (nibble (x.Q (k/8) n) k-nibble (x.Z (k/c.group) (n/8)) n)*x.S (k/c.group) n
def mathematical (c : Call) (x : Input) (r n : Nat) : ℝ :=
  ∑ k∈Finset.range c.K,x.A r k*dequant c x k n
/-- Source's two products and subtraction, with the actual group branch,
packed K addressing and reused lane shifter. Casts are identities only here. -/
def sourceB (c : Call) (i : Fin 17) (x : Input) (step t n : Nat) : ℝ :=
  let p := params c i
  let g := groupIndex p step
  nibble (x.Q (step*(p.blockInner/8)+t/8) n) t*x.S g n-
    nibble (x.Z g (n/8)) n*x.S g n

def sourceTerm (c : Call) (i : Fin 17) (x : Input) (r n step t : Nat) : ℝ :=
  x.A r (step*(tunings i).BK+t)*sourceB c i x step t n
/-- Actual nested K-block/K16 summation schedule interpreted over reals. -/
def accumulator (c : Call) (i : Fin 17) (x : Input) (r n : Nat) : Nat → ℝ
  | 0 => 0
  | step+1 => accumulator c i x r n step+
      chunks (sourceTerm c i x r n step) 16 ((tunings i).BK/16)
def value (c : Call) (i : Fin 17) (x : Input) (r n : Nat) := accumulator c i x r n (steps (params c i))

theorem sourceB_global (c : Call) (h : c.Valid) (i : Fin 17) (x : Input)
    (step : Nat) (hs : step<steps (params c i)) (t : Fin (tunings i).BK) (n : Nat) :
    sourceB c i x step t.val n=dequant c x (step*(tunings i).BK+t.val) n := by
  have hg := Logical.group_eq (params c i) (legal c h i) step hs t
  have hd := Logical.packed_div (params c i) (legal c h i) step t.val
  have hm := Logical.packed_mod (params c i) (legal c h i) step t.val
  change groupIndex (params c i) step=(step*(tunings i).BK+t.val)/c.group at hg
  change (step*(tunings i).BK+t.val)/8=step*((tunings i).BK/8)+t.val/8 at hd
  change (step*(tunings i).BK+t.val)%8=t.val%8 at hm
  dsimp only [sourceB]
  rw [hg]
  simp only [dequant,nibble,hd,hm]
  change _ = _
  dsimp only [params]
  ring

theorem accumulator_global (c : Call) (h : c.Valid) (i : Fin 17) (x : Input) (r n count : Nat)
    (hc : count≤steps (params c i)) :
    accumulator c i x r n count=chunks (fun k => x.A r k*dequant c x k n) (tunings i).BK count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [accumulator,chunks,ih (by omega)]
    simp only [chunks_eq,Nat.div_mul_cancel (tuning_facts i).2.2.2.2.2.2]
    congr 1
    apply sum_congr
    intro t ht
    unfold sourceTerm
    rw [sourceB_global c h i x count (by omega) ⟨t,ht⟩ n]

theorem exact_schedule (c : Call) (h : c.Valid) (i : Fin 17) (x : Input) (r n : Nat) :
    value c i x r n=mathematical c x r n := by
  rw [value,accumulator_global c h i x r n _ (by omega),chunks_eq]
  have hk := (legal c h i).fullK
  change steps (params c i)*(tunings i).BK=c.K at hk
  rw [hk,sum_finset]
  rfl

#specNonCircular mathematical avoiding [value,accumulator,sourceTerm,sourceB,Logical.output,Native.grid]
#axiomsClean sourceB_global
#axiomsClean accumulator_global
#axiomsClean exact_schedule
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Real
