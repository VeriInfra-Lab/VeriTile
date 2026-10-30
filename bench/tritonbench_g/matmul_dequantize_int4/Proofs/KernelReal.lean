import bench.tritonbench_g.matmul_dequantize_int4.Proofs.RealHost
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Mathematical_FPComplete

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4.FPTyped
open MatmulKernel.RealSum
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- Real interpretation of the operations reached by the selected shared DSL.
Destination casts are identities only in this mathematical interpretation. -/
def scalar : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | _ => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ x => x
  fromInt := fun _ x => x
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ _ => 0

def dotValue {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) :
    TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ e : Fin k, a (i.1,e,PUnit.unit)*b (e,i.2.1,PUnit.unit)
  | _::tail => fun i => dotValue (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def dot : Dot ℝ := ⟨fun _ _ => dotValue⟩

/-- Ordinary signed-int32 representation of the two packed Tensor inputs. -/
def Packed (s : State ℝ) : Prop :=
  ∀ region, region="Q" ∨ region="Z" → ∀ offset,
    ∃ word : BitVec 32, s.mem region offset=.mk .int word.toInt

theorem nibble_real (word : BitVec 32) (slot : Nat) :
    (MatmulDequantInt4.FPTyped.nibble word.toInt slot : ℝ)=Real.nibble word slot := by
  rw [MatmulDequantInt4.FPTyped.nibble_packed_word]
  have bound : (((word >>> (slot%8*4)) &&& (15 : BitVec 32)).toNat)≤15 := by
    simp only [BitVec.toNat_and]
    exact Nat.and_le_right
  rw [BitVec.toInt_eq_toNat_of_lt (by omega)]
  simp only [Real.nibble,Packing.nibble,BitVec.extractLsb'_toNat,
    BitVec.toNat_and,BitVec.toNat_ushiftRight]
  norm_num only [BitVec.toNat_ofNat,Int.cast_natCast]
  congr 1
  exact Nat.and_two_pow_sub_one_eq_mod _ 4

theorem a_source (c : Call) (i : Fin 17) (s : State ℝ) (step : Nat)
    (lane : AIndex (params c i)) (live : inputRow (params c i) s lane.1<c.M) :
    aValue scalar (params c i) s step lane=
      (RealHost.input c s).A (inputRow (params c i) s lane.1)
        (step*(tunings i).BK+lane.2.1.val) := by
  unfold aValue
  change (if inputRow (params c i) s lane.1<c.M then _ else _)=_
  rw [if_pos live]
  change (s.mem "A" (inputRow (params c i) s lane.1*c.A.sr+
    lane.2.1.val*c.A.sc+step*((tunings i).BK*c.A.sc))).read .fp16=
    (s.mem "A" (c.A.addr (inputRow (params c i) s lane.1)
      (step*(tunings i).BK+lane.2.1.val))).read .fp16
  apply congrArg (fun offset => (s.mem "A" offset).read .fp16)
  unfold MatmulKernel.View.addr
  ring

theorem b_source (c : Call) (i : Fin 17) (s : State ℝ) (packed : Packed s)
    (step : Nat) (lane : BIndex (params c i)) :
    bValue scalar (params c i) s step lane=
      Real.sourceB c i (RealHost.input c s) step lane.1.val
        (inputColumn (params c i) s lane.2.1) := by
  let p := params c i
  let n := inputColumn p s lane.2.1
  have qoffset : bOffset p s step lane=
      c.Q.addr (step*(p.blockInner/8)+lane.1.val/8) n := by
    simp only [bOffset,inputInner,MatmulKernel.View.addr,p,params,n]
    ring
  have soffset : scaleOffset p s step (lane.2.1,PUnit.unit)=
      c.S.addr (groupIndex p step) n := by
    simp only [scaleOffset,MatmulKernel.View.addr,p,params,n]
    ring
  have zoffset : zeroOffset p s step (lane.2.1,PUnit.unit)=
      c.Z.addr (groupIndex p step) (n/8) := by
    simp only [zeroOffset,MatmulKernel.View.addr,p,params,n]
    ring
  obtain ⟨q,hq⟩ := packed "Q" (Or.inl rfl) (bOffset p s step lane)
  obtain ⟨z,hz⟩ := packed "Z" (Or.inr rfl) (zeroOffset p s step (lane.2.1,PUnit.unit))
  change bValue scalar p s step lane=Real.sourceB c i (RealHost.input c s) step lane.1.val n
  simp only [bValue,zeroValue,scalar,weightNibble,zeroNibble,scaleValue,
    show p.weight="Q" from rfl,show p.zero="Z" from rfl,show p.scale="S" from rfl,
    hq,hz,Cell.read_same,nibble_real]
  simp only [Real.sourceB,RealHost.input]
  rw [←qoffset,←zoffset,←soffset,hq,hz]
  simp only [Cell.read_same,BitVec.ofInt_toInt]
  rfl

theorem block_sum (c : Call) (i : Fin 17) (s : State ℝ) (packed : Packed s)
    (step : Nat) (lane : CIndex (params c i))
    (live : inputRow (params c i) s lane.1<c.M) :
    dot.apply (some .fp32) {} (batch:=[]) (aValue scalar (params c i) s step)
      (fun j => scalar.cast (some .fp32) .fp16 .fp16 (bValue scalar (params c i) s step j)) lane=
    sum (Real.sourceTerm c i (RealHost.input c s)
      (inputRow (params c i) s lane.1) (inputColumn (params c i) s lane.2.1) step) (tunings i).BK := by
  have cast_half (x : ℝ) : scalar.cast (some .fp32) .fp16 .fp16 x=x := rfl
  simp only [dot,dotValue,cast_half]
  rw [sum_finset,←Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro e _
  unfold Real.sourceTerm
  rw [a_source c i s step (lane.1,e,PUnit.unit) live,
    b_source c i s packed step (e,lane.2.1,PUnit.unit)]

theorem accumulation_source (c : Call) (i : Fin 17) (s : State ℝ) (packed : Packed s)
    (lane : CIndex (params c i)) (live : inputRow (params c i) s lane.1<c.M) (count : Nat) :
    accumulation scalar dot (params c i) s count lane=
      Real.accumulator c i (RealHost.input c s)
        (inputRow (params c i) s lane.1) (inputColumn (params c i) s lane.2.1) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    change accumulation scalar dot (params c i) s count lane+
      dot.apply (some .fp32) {} (batch:=[]) (aValue scalar (params c i) s count)
        (fun j => scalar.cast (some .fp32) .fp16 .fp16 (bValue scalar (params c i) s count j)) lane=_
    rw [ih,block_sum c i s packed count lane live]
    simp only [Real.accumulator,chunks_eq,Nat.div_mul_cancel (tuning_facts i).2.2.2.2.2.2]

theorem output_mathematical (c : Call) (valid : c.Valid) (i : Fin 17)
    (s : State ℝ) (packed : Packed s) (lane : CIndex (params c i))
    (live : active (params c i) s lane) :
    outputValue scalar dot (params c i) s lane=
      Real.mathematical c (RealHost.input c s)
        (inputRow (params c i) s lane.1) (inputColumn (params c i) s lane.2.1) := by
  change accumulation scalar dot (params c i) s (steps (params c i)) lane=_
  rw [accumulation_source c i s packed lane live.1]
  exact Real.exact_schedule c valid i (RealHost.input c s) _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelReal
