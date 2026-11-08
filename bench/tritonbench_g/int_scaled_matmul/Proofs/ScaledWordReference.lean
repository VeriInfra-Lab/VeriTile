import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordSource
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
local instance scaledReferenceDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State Word) := s.pids 0 / (p.groupRows*blocksN p) * p.groupRows
def rowsInGroup (p : Parameters) (s : State Word) := min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State Word) := firstRow p s + s.pids 0 % rowsInGroup p s
def columnBlock (p : Parameters) (s : State Word) := (s.pids 0 % (p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State Word) (i : Fin p.blockRows) := (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := (columnBlock p s*p.blockColumns+j.val)%p.columns
def inputInner (p : Parameters) (_s : State Word) (j : Fin p.blockInner) := j.val
def aOffset (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+step*(p.blockInner*p.inputInnerStride)
def bOffset (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) :=
  inputInner p s i.1*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+step*(p.blockInner*p.weightInnerStride)
def aValue (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.2.1 < p.inner-step*p.blockInner then
    (s.mem p.input (aOffset p s step i)).read .int else 0)
def bValue (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Int :=
  wrap 8 (if p.even=Bool.true ∨ inputInner p s i.1 < p.inner-step*p.blockInner then
    (s.mem p.weight (bOffset p s step i)).read .int else 0)
def accumulation (p : Parameters) (s : State Word) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => wrap 32 (accumulation p s k i + (FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i).toInt)

def outputRow (p : Parameters) (s : State Word) (i : CIndex p) := rowBlock p s*p.blockRows+i.1.val
def outputColumn (p : Parameters) (s : State Word) (i : CIndex p) := columnBlock p s*p.blockColumns+i.2.1.val
/-- The repaired suffix uses the actual output and scale row strides. -/
def outputOffset (p : Parameters) (s : State Word) (i : CIndex p) :=
  outputColumn p s i*p.outputColumnStride+outputRow p s i*p.outputRowStride
def active (p : Parameters) (s : State Word) (i : CIndex p) : Prop :=
  outputRow p s i < p.rows ∧ outputColumn p s i < p.columns
instance (p : Parameters) (s : State Word) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
/-- No other=0 is inserted for the original scale load. The execution may
use its typed undefined value on an inactive lane; observed values below
read only the actual initial scale row. -/
def rawScale (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  if active p s i then (s.mem p.scale (outputRow p s i*p.scaleRowStride)).read .real
  else s.undef .real p.scale (outputRow p s i*p.scaleRowStride)
def scaled (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  FP.mul cfg (FP.round .fp32 cfg (accumulation p s p.steps i : Rat))
    ((s.mem p.scale (outputRow p s i*p.scaleRowStride)).read .real)
def expected (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Int :=
  (toSigned 32 (scaled cfg p s i)).getD 0
def InputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p s i → (toSigned 32 (scaled cfg p s i)).isSome
@[simp] theorem wrap_accumulation (p : Parameters) (s : State Word) (k : Nat) (i : CIndex p) :
    wrap 32 (accumulation p s k i)=accumulation p s k i := by
  cases k <;> simp [accumulation,wrap,BitVec.ofInt_toInt]

/-- The source stops at K, even when the historical proof parameter admits
extra masked iterations. Those iterations carry only zero products. -/
theorem stopped_accumulation (p : Parameters) (s : State Word)
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (k : Nat) (hk : k≤p.steps) (stopped : p.inner≤k*p.blockInner) (i : CIndex p) :
    accumulation p s p.steps i=accumulation p s k i := by
  by_cases zero : p.blockInner=0
  · have empty : List.finRange p.blockInner=[] := by
      apply List.eq_nil_of_length_eq_zero
      simpa using zero
    have allzero (n : Nat) : accumulation p s n i=0 := by
      induction n with
      | zero=>rfl
      | succ n ih=>simp [accumulation,ih,FP.IntegerDot.wordDot,empty,wrap]
    rw [allzero,allzero]
  by_cases even : p.even=Bool.true
  · have same : k=p.steps := by
      have hx:=heven even
      have positive : 0<p.blockInner := by omega
      nlinarith
    rw [same]
  have false_flag : p.even=Bool.false := by cases h:p.even <;> simp_all
  have zero_dot (n : Nat) (hn : k≤n) :
      (FP.IntegerDot.wordDot [] (aValue p s n) (bValue p s n) i).toInt=0 := by
    have bound : p.inner≤n*p.blockInner := stopped.trans (Nat.mul_le_mul_right _ hn)
    have az : aValue p s n=(fun _=>0) := by
      funext j
      simp [aValue,false_flag,Nat.sub_eq_zero_of_le bound,wrap]
    have bz : bValue p s n=(fun _=>0) := by
      funext j
      simp [bValue,false_flag,Nat.sub_eq_zero_of_le bound,wrap]
    rw [az,bz,Word32.dot_zero]
  have stable (n : Nat) : k≤n → accumulation p s n i=accumulation p s k i := by
    induction n with
    | zero=>
      intro h
      have same : k=0 := by omega
      subst k
      rfl
    | succ n ih=>
      intro hn
      by_cases eq : k=n+1
      · rw [eq]
      have less : k≤n := by omega
      simp only [accumulation]
      rw [zero_dot n less,Int.add_zero,wrap_accumulation,ih less]
  exact stable p.steps hk

end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
