import bench.tritonbench_g.int8_matmul_quantization.Proofs.MatmulComplete
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance matmulContractDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Independent unbounded sum used only as a specification: the executable
backend uses finite words. The value theorem projects this sum to signed32
before any floating conversion, including when the sum overflows int32. -/
def blockSum (p : Parameters) (s : State Word) (k : Nat) (i : CIndex p) : Int :=
  (List.finRange p.blockInner).foldl
    (fun total t => total + aValue p s k (i.1,t,PUnit.unit)*bValue p s k (t,i.2.1,PUnit.unit)) 0
def exactAccumulation (p : Parameters) (s : State Word) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => exactAccumulation p s k i + blockSum p s k i

theorem dot_low32 (p : Parameters) (s : State Word) (k : Nat) (i : CIndex p) :
    FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i = BitVec.ofInt 32 (blockSum p s k i) := by
  rcases i with ⟨r,c,u⟩
  cases u
  exact FP.IntegerDot.fold_is_low32 _ _ _ 0

theorem accumulation_low32 (p : Parameters) (s : State Word) (n : Nat) (i : CIndex p) :
    BitVec.ofInt 32 (accumulation p s n i) = BitVec.ofInt 32 (exactAccumulation p s n i) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [accumulation, wrap, BitVec.ofInt_toInt, exactAccumulation, BitVec.ofInt_add]
    rw [ih, dot_low32]

theorem accumulation_value (p : Parameters) (s : State Word) (n : Nat) (i : CIndex p) :
    accumulation p s n i = wrap 32 (exactAccumulation p s n i) := by
  have h := congrArg BitVec.toInt (accumulation_low32 p s n i)
  cases n with
  | zero => rfl
  | succ n =>
    simpa only [accumulation, wrap, BitVec.ofInt_toInt] using h

def exactOutput (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow cfg
    (FP.mul cfg (FP.mul cfg (FP.round .fp32 cfg (wrap 32 (exactAccumulation p s p.steps i) : Rat))
      (QuantizeKvTransform.MixedBits.widen (aScale p s i.1))) (bScale p s i.2.1))

theorem output_spec (cfg : FP.Config) (p : Parameters) (s : State Word) :
    outputValue cfg p s = exactOutput cfg p s := by
  funext i
  unfold outputValue exactOutput
  rw [accumulation_value]

/-- Every original K position has exactly its quotient/remainder block/lane
representation and an active source mask. Extra blocks remain permitted. -/
theorem inner_coverage (p : Parameters) (s : State Word)
    (hk : 0<p.blockInner) (hcover : p.inner ≤ p.steps*p.blockInner) (hpid : s.pids 1=0)
    (j : Fin p.inner) :
    ∃ k, ∃ lane : Fin p.blockInner, k<p.steps ∧
      inputInner p s lane < p.inner-k*p.blockInner ∧
      k*p.blockInner+inputInner p s lane=j.val := by
  let lane : Fin p.blockInner := ⟨j.val%p.blockInner, Nat.mod_lt _ hk⟩
  have hdiv : j.val/p.blockInner < p.steps := (Nat.div_lt_iff_lt_mul hk).mpr (lt_of_lt_of_le j.isLt hcover)
  have heq : j.val/p.blockInner*p.blockInner+j.val%p.blockInner=j.val := Nat.div_add_mod' _ _
  refine ⟨j.val/p.blockInner,lane,hdiv,?_,?_⟩ <;> simp only [inputInner,hpid,zero_mul,zero_add,lane]
  · omega
  · exact heq

theorem machine_correctness (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i=outputOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (backend rest cfg) (source p) initial=some final ∧
      Post p initial final (exactOutput cfg p initial) := by
  simpa only [output_spec] using correctness rest cfg p initial hw

#axiomsClean dot_low32
#axiomsClean accumulation_low32
#axiomsClean accumulation_value
#axiomsClean output_spec
#axiomsClean inner_coverage
#axiomsClean machine_correctness
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
