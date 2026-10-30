import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOData

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOMath
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4.FPTyped
open MatmulKernel.RealSum
noncomputable section
set_option maxHeartbeats 2000000

def formula (c : RawKernel.Layout) (x : ChannelValues ℝ (IOData.inputs c)) :=
  IOData.expected c (Real.mathematical c.call (input c x))

theorem packed_k (c : RawKernel.Layout) (h : c.Valid) (k : Nat) (hk : k<c.K) :
    k/8<c.K/8 := by
  apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
  simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣128) h.k128)] using hk

theorem packed_n (c : RawKernel.Layout) (h : c.Valid) (n : Nat) (hn : n<c.N) :
    n/8<c.N/8 := by
  apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
  simpa only [Nat.div_mul_cancel (dvd_trans (by decide : 8∣256) h.n256)] using hn

theorem b_source (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation ℝ)) (x : ChannelValues ℝ (IOData.inputs c))
    (packed : IOData.packedInt32 c x) (s : State ℝ)
    (binding : (IOData.program ℝ c jobs).InputBinding x s)
    (step : Nat) (hs : step<steps (params c.call i)) (lane : BIndex (params c.call i))
    (hn : inputColumn (params c.call i) s lane.2.1<c.N) :
    bValue KernelReal.scalar (params c.call i) s step lane=
      Real.sourceB c.call i (input c x) step lane.1.val
        (inputColumn (params c.call i) s lane.2.1) := by
  let p := params c.call i
  let n := inputColumn p s lane.2.1
  let k := step*p.blockInner+lane.1.val
  have hk : k<c.K := Logical.k_bound p (legal c.call (c.valid valid) i) step hs lane.1
  have hg := Bounds.group_bound c.call (c.valid valid) k hk
  have hq := packed_k c valid k hk
  have hz := packed_n c valid n hn
  have kd := Logical.packed_div p (legal c.call (c.valid valid) i) step lane.1.val
  have gd := Logical.group_eq p (legal c.call (c.valid valid) i) step hs lane.1
  change k/8=step*(p.blockInner/8)+lane.1.val/8 at kd
  change groupIndex p step=k/c.group at gd
  change k/c.group<c.K/c.group at hg
  have eqQ : bOffset p s step lane=c.Q.addr (k/8) n := by
    change lane.1.val/8*c.Q.sr+n*c.Q.sc+step*(p.blockInner/8*c.Q.sr)=
      k/8*c.Q.sr+n*c.Q.sc
    rw [kd]
    ring
  have eqS : scaleOffset p s step (lane.2.1,())=c.S.addr (k/c.group) n := by
    change n*c.S.sc+groupIndex p step*c.S.sr=k/c.group*c.S.sr+n*c.S.sc
    rw [gd]
    omega
  have eqZ : zeroOffset p s step (lane.2.1,())=c.Z.addr (k/c.group) (n/8) := by
    change n/8*c.Z.sc+groupIndex p step*c.Z.sr=k/c.group*c.Z.sr+n/8*c.Z.sc
    rw [gd]
    omega
  obtain ⟨qw,hqw⟩ := packed.1 (⟨k/8,hq⟩,⟨n,hn⟩,())
  obtain ⟨zw,hzw⟩ := packed.2 (⟨k/c.group,hg⟩,⟨n/8,hz⟩,())
  have aq : (IOData.arrays c x).Q (k/8) n=qw.toInt := by
    change (if h : k/8<c.K/8 ∧ n<c.N then x (⟨1,by decide⟩ : Fin 4) (⟨k/8,h.1⟩,⟨n,h.2⟩,()) else (0 : Int))=qw.toInt
    rw [dif_pos (show k/8<c.K/8 ∧ n<c.N from ⟨hq,hn⟩)]
    exact hqw
  have az : (IOData.arrays c x).Z (k/c.group) (n/8)=zw.toInt := by
    change (if h : k/c.group<c.K/c.group ∧ n/8<c.N/8 then x (⟨3,by decide⟩ : Fin 4) (⟨k/c.group,h.1⟩,⟨n/8,h.2⟩,()) else (0 : Int))=zw.toInt
    rw [dif_pos (show k/c.group<c.K/c.group ∧ n/8<c.N/8 from ⟨hg,hz⟩)]
    exact hzw
  have rq := (IOData.binding_q c jobs x s binding _ _ hq hn).trans aq
  have rz := (IOData.binding_z c jobs x s binding _ _ hg hz).trans az
  have rs := IOData.binding_s c jobs x s binding _ _ hg hn
  change (s.mem "Q" (c.Q.addr (k/8) n)).read .int=qw.toInt at rq
  change (s.mem "Z" (c.Z.addr (k/c.group) (n/8))).read .int=zw.toInt at rz
  change (s.mem "S" (c.S.addr (k/c.group) n)).read .fp16=(IOData.arrays c x).S (k/c.group) n at rs
  change bValue KernelReal.scalar p s step lane=Real.sourceB c.call i (input c x) step lane.1.val n
  simp only [bValue,zeroValue,weightNibble,zeroNibble,scaleValue,eqQ,eqS,eqZ,
    show p.weight="Q" from rfl,show p.zero="Z" from rfl,show p.scale="S" from rfl,rq,rz,rs,
    KernelReal.scalar,KernelReal.nibble_real]
  simp only [Real.sourceB,input]
  change _ = Real.nibble (BitVec.ofInt 32 ((IOData.arrays c x).Q
      (step*(p.blockInner/8)+lane.1.val/8) n)) lane.1.val *
      (IOData.arrays c x).S (groupIndex p step) n -
    Real.nibble (BitVec.ofInt 32 ((IOData.arrays c x).Z (groupIndex p step) (n/8))) n *
      (IOData.arrays c x).S (groupIndex p step) n
  rw [←kd,gd,aq,az,BitVec.ofInt_toInt,BitVec.ofInt_toInt]

theorem block_sum (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation ℝ)) (x : ChannelValues ℝ (IOData.inputs c))
    (packed : IOData.packedInt32 c x) (s : State ℝ)
    (binding : (IOData.program ℝ c jobs).InputBinding x s)
    (step : Nat) (hs : step<steps (params c.call i)) (lane : CIndex (params c.call i))
    (live : active (params c.call i) s lane) :
    KernelReal.dot.apply (some .fp32) {} (batch:=[])
      (aValue KernelReal.scalar (params c.call i) s step)
      (fun j => KernelReal.scalar.cast (some .fp32) .fp16 .fp16
        (bValue KernelReal.scalar (params c.call i) s step j)) lane=
    sum (Real.sourceTerm c.call i (input c x)
      (inputRow (params c.call i) s lane.1) (inputColumn (params c.call i) s lane.2.1) step)
      (tunings i).BK := by
  have cast_half (v : ℝ) : KernelReal.scalar.cast (some .fp32) .fp16 .fp16 v=v := rfl
  simp only [KernelReal.dot,KernelReal.dotValue,cast_half]
  rw [sum_finset,←Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro e _
  unfold Real.sourceTerm
  rw [KernelReal.a_source c.call i s step (lane.1,e,()) live.1,
    b_source c valid i jobs x packed s binding step hs (e,lane.2.1,()) live.2]
  congr 1
  exact IOData.binding_a c jobs x s binding _ _ live.1
    (Logical.k_bound _ (legal c.call (c.valid valid) i) step hs e)

theorem accumulation (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation ℝ)) (x : ChannelValues ℝ (IOData.inputs c))
    (packed : IOData.packedInt32 c x) (s : State ℝ)
    (binding : (IOData.program ℝ c jobs).InputBinding x s)
    (lane : CIndex (params c.call i)) (live : active (params c.call i) s lane)
    (count : Nat) (hc : count≤steps (params c.call i)) :
    MatmulDequantizeInt4.FPTyped.accumulation KernelReal.scalar KernelReal.dot (params c.call i) s count lane=
      Real.accumulator c.call i (input c x)
        (inputRow (params c.call i) s lane.1) (inputColumn (params c.call i) s lane.2.1) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    change MatmulDequantizeInt4.FPTyped.accumulation KernelReal.scalar KernelReal.dot (params c.call i) s count lane+
      KernelReal.dot.apply (some .fp32) {} (batch:=[])
        (aValue KernelReal.scalar (params c.call i) s count)
        (fun j => KernelReal.scalar.cast (some .fp32) .fp16 .fp16
          (bValue KernelReal.scalar (params c.call i) s count j)) lane=_
    rw [ih (by omega),block_sum c valid i jobs x packed s binding count (by omega) lane live]
    simp only [Real.accumulator,chunks_eq,Nat.div_mul_cancel (tuning_facts i).2.2.2.2.2.2]

theorem output (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17)
    (jobs : List (Invocation ℝ)) (x : ChannelValues ℝ (IOData.inputs c))
    (packed : IOData.packedInt32 c x) (s : State ℝ)
    (binding : (IOData.program ℝ c jobs).InputBinding x s)
    (lane : CIndex (params c.call i)) (live : active (params c.call i) s lane) :
    outputValue KernelReal.scalar KernelReal.dot (params c.call i) s lane=
      Real.mathematical c.call (input c x)
        (inputRow (params c.call i) s lane.1) (inputColumn (params c.call i) s lane.2.1) := by
  change MatmulDequantizeInt4.FPTyped.accumulation KernelReal.scalar KernelReal.dot _ _ _ _=_
  rw [accumulation c valid i jobs x packed s binding lane live _ (by omega)]
  exact Real.exact_schedule c.call (c.valid valid) i (input c x) _ _

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOMath
