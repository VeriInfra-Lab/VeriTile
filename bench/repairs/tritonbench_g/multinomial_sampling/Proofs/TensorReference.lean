import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorInputs

namespace VeriTile.Bench.Repairs.MultinomialSampling.TensorReference
open VeriTile.Triton FP.Structural SourceValue TensorInputs
noncomputable section
variable {α : Type} [Inhabited α]

/-- Input-only logical Tensor values and the ordinary undefined masked-load
payloads. There is no executor or memory-state result in this recurrence. -/
structure Data (α : Type) where
  pid : Nat
  score : Nat → Nat → α
  label : Nat → Nat → Int
  seed : Nat → Int
  offset : Nat → Int
  undefined : RegionName → Nat → Int

def logicalData (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (pid : Nat) (undefined : RegionName → Nat → Int) : Data α :=
  ⟨pid,fun r j => interpret (scoreWord p input bytes r j),
    fun r j => (labelWord p input bytes r j).toInt,
    fun r => (seedWord p input bytes r).toInt,
    fun r => (offsetWord p input bytes r).toInt,undefined⟩

def row (p : Parameters) (d : Data α) (i : Row p) := d.pid*p.block+i.1.val

def sample (P : Primitives α) (p : Parameters) (d : Data α) (i : Row p) : α :=
  let r := row p d i
  let seed := if r<p.batches then d.seed r else d.undefined p.seeds r
  let offset := if r<p.batches then d.offset r else d.undefined p.offsets r
  P.bits32 ⟨(Random.rand seed (BitVec.ofInt 32 offset).toInt .bits32 10).bits⟩

def scores (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat) (i : Lane p) : α :=
  P.cast (some .fp32) .real .real
    (if row p d (i.1,PUnit.unit)<p.batches ∧ b+i.2.1.val<p.tokens
      then d.score (row p d (i.1,PUnit.unit)) (b+i.2.1.val) else padding P .fp32)

def scanned (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat) : Lane p → α :=
  P.scan none .sum ⟨1,by simp⟩ .forward (scores P p d b)

def cumulative (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat)
    (st : Accumulator α p) (i : Lane p) : α :=
  P.add (st.acc (i.1,PUnit.unit)) (scanned P p d b i)
def valid (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat)
    (st : Accumulator α p) (i : Lane p) : Bool :=
  P.lt (P.sub (cumulative P p d b st i) (scores P p d b i)) (sample P p d (i.1,PUnit.unit)) &&
    P.le (sample P p d (i.1,PUnit.unit)) (cumulative P p d b st i)
def found (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat)
    (st : Accumulator α p) (i : Row p) : Bool :=
  decide (0<TileShape.discreteBoolSum ⟨1,by simp⟩ Bool.false (valid P p d b st) i)
def selected (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat)
    (st : Accumulator α p) (i : Row p) : Nat :=
  b+TileShape.discreteIntArgMax ⟨1,by simp⟩ (fun j => if valid P p d b st j then (1:Int) else 0) i

def advance (P : Primitives α) (p : Parameters) (d : Data α) (b : Nat)
    (st : Accumulator α p) : Accumulator α p where
  acc := fun i => P.add (st.acc i) (P.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned P p d b) i)
  output := fun i => if found P p d b st i then
    (if row p d i<p.batches then d.label (row p d i) (min (selected P p d b st i) (p.tokens-1)) else -1)
    else st.output i

def initial (P : Primitives α) (p : Parameters) (d : Data α) : Accumulator α p :=
  ⟨fun _ => P.zero (some .fp32) .real,fun i =>
    if row p d i<p.batches then d.label (row p d i) 0
      else d.undefined p.indices (row p d i*p.strideIndexBatch)⟩
def rounds (P : Primitives α) (p : Parameters) (d : Data α) : Nat → Accumulator α p
  | 0 => initial P p d
  | k+1 => advance P p d (k*p.blockTokens) (rounds P p d k)

def written (P : Primitives α) (p : Parameters) (d : Data α) (s : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i =>
    if row p d i<p.batches then t.write p.output (row p d i)
      (.mk .int ((rounds P p d (count p)).output i)) else t) s

variable (P : Primitives α) (p : Parameters) (input : Inputs p) (bytes : Bytes)
  (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s)
local notation "d" => logicalData p input bytes interpret (s.pids 0) (s.undef TileDType.int)

include loaded

theorem scores_eq (b : Nat) : SourceValue.scores P .fp32 p s b=scores P p d b := by
  funext i
  simp only [SourceValue.scores,scores,readScore,Storage.dtype,row,logicalData,
    SourceValue.active,SourceValue.row,scoreOffset]
  split_ifs with h
  · exact congrArg (P.cast (some .fp32) .real .real)
      (score_read p input bytes interpret s loaded _ _ h.1 h.2)
  · rfl

theorem sample_eq : SourceValue.sample P p s=sample P p d := by
  funext i
  simp only [SourceValue.sample,seedValue,offsetValue,intLoad,sample,row,logicalData,
    SourceValue.active,SourceValue.row]
  by_cases ha : s.pids 0*p.block+i.1.val<p.batches
  · simp [ha,seed_read p input bytes interpret s loaded _ ha,
      offset_read p input bytes interpret s loaded _ ha]
  · simp [ha]

theorem valid_eq (b : Nat) (st : Accumulator α p) :
    SourceValue.valid P .fp32 p s b st=valid P p d b st := by
  funext i
  simp only [SourceValue.valid,SourceValue.cumulative,SourceValue.scanned,valid,cumulative,scanned,
    scores_eq P p input bytes interpret s loaded,sample_eq P p input bytes interpret s loaded]

theorem rounds_eq (positive : 0<p.tokens) (k : Nat) :
    SourceValue.rounds P .fp32 p s k=rounds P p d k := by
  induction k with
  | zero =>
    unfold SourceValue.rounds SourceValue.initialAccumulator rounds initial
    congr 1
    funext i
    simp only [SourceValue.initialOutput,intLoad,SourceValue.active,SourceValue.row,row,logicalData]
    by_cases ha : s.pids 0*p.block+i.1.val<p.batches
    · have h := label_read p input bytes interpret s loaded _ 0 ha positive
      simpa [ha] using h
    · simp [ha]
  | succ k ih =>
    unfold SourceValue.rounds rounds
    rw [ih]
    unfold SourceValue.advance advance
    congr 1
    · funext i
      simp only [SourceValue.scanned,scanned,scores_eq P p input bytes interpret s loaded]
    · funext i
      have hv := valid_eq P p input bytes interpret s loaded (k*p.blockTokens) (rounds P p d k)
      have hfound : SourceValue.found P .fp32 p s (k*p.blockTokens) (rounds P p d k)=
          found P p d (k*p.blockTokens) (rounds P p d k) := by
        funext j
        unfold SourceValue.found TensorReference.found
        rw [hv]
      have hselected : SourceValue.selected P .fp32 p s (k*p.blockTokens) (rounds P p d k)=
          selected P p d (k*p.blockTokens) (rounds P p d k) := by
        funext j
        unfold SourceValue.selected TensorReference.selected
        rw [hv]
      rw [hfound]
      by_cases hf : found P p d (k*p.blockTokens) (rounds P p d k) i=Bool.true
      · simp only [hf,↓reduceIte,SourceValue.selectedIndex,hfound,Bool.true_and,hselected]
        by_cases ha : SourceValue.active p s i
        · have bound := KernelMemory.clamp_active p.tokens (selected P p d (k*p.blockTokens) (rounds P p d k) i) positive
          have hl := label_read p input bytes interpret s loaded _ _ ha bound
          have har : s.pids 0*p.block+i.1.val<p.batches := ha
          simpa [ha,har,logicalData,row,SourceValue.row] using hl
        · have har : ¬row p d i<p.batches := ha
          simp [ha,har]
      · simp [hf]

theorem written_eq (positive : 0<p.tokens) :
    (SourceValue.written P .fp32 p s).mem=(written P p d s).mem := by
  simp only [SourceValue.written,written,rounds_eq P p input bytes interpret s loaded positive,
    SourceValue.active,SourceValue.row,row,logicalData]

#specNonCircular rounds avoiding [SourceValue.rounds,SourceValue.written,source,FP.Structural.exec]
#axiomsClean scores_eq
#axiomsClean sample_eq
#axiomsClean rounds_eq
#axiomsClean written_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.TensorReference
