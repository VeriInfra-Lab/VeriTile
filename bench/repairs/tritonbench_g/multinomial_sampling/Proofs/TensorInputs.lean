import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.MultinomialSampling.TensorInputs
open VeriTile.Triton FP.Structural SourceValue KernelMemory
noncomputable section

/-- A single entry-memory byte reader serves all input tensors, including
aliases and expanded labels. Scalar words use the selected little-endian ABI. -/
abbrev Bytes := Nat → BitVec 8

def readWord (bytes : Bytes) (n address : Nat) : BitVec (8*n) :=
  BitVec.ofNat (8*n) ((List.range n).foldl (fun value i =>
    value+(bytes (address+i)).toNat*2^(8*i)) 0)
def read32 (bytes : Bytes) (address : Nat) : VeriTile.Torch.FP32.Word :=
  ⟨readWord bytes 4 address⟩
def read64 (bytes : Bytes) (address : Nat) : BitVec 64 := readWord bytes 8 address

def encode (word : VeriTile.Torch.FP32.Word) : BitVec 64 := word.bits.zeroExtend 64
/-- Explicit mathematical interpretation; finite readback below supplies the
actual Real denotation and excludes exceptional score words through the guard. -/
def denote (word : VeriTile.Torch.FP32.Word) : ℝ := (word.toRat?.getD 0 : ℝ)
def wordToReal (word : BitVec 64) : ℝ := denote ⟨word.truncate 32⟩

theorem wordToReal_encode (word : VeriTile.Torch.FP32.Word) : wordToReal (encode word)=denote word := by
  unfold wordToReal encode
  apply congrArg denote
  cases word with
  | mk bits =>
    congr 1
    change (bits.setWidth 64).setWidth 32=bits
    rw [BitVec.setWidth_setWidth_of_le bits (by decide : 32 ≤ 64)]
    exact BitVec.setWidth_eq bits

structure IntegerView (rows columns sb st : Nat) where
  tensor : IntegerTensor
  laws : IntegerLaws tensor
  ordinary : tensor.metadata.isNeg=Bool.false
  rowCount : tensor.rows=rows
  columnCount : tensor.columns=columns
  batchStride : tensor.strideBatch=sb
  tokenStride : tensor.strideToken=st

/-- Same-call Tensor descriptors and ordinary layout/stride observations.
Vectors use a singleton second coordinate; its zero stride is unused. -/
structure Inputs (p : Parameters) where
  scores : VeriTile.Torch.FP32.Tensor
  scoreLaws : Scores p scores
  labels : IntegerView p.batches p.tokens p.strideIndexBatch p.strideIndexToken
  seeds : IntegerView p.batches 1 1 0
  offsets : IntegerView p.batches 1 1 0

def scoreWord (p : Parameters) (input : Inputs p) (bytes : Bytes) (r j : Nat) : VeriTile.Torch.FP32.Word :=
  VeriTile.Torch.FP32.logicalValue (read32 bytes) input.scores input.scoreLaws.laws (r*p.tokens+j)
def labelWord (p : Parameters) (input : Inputs p) (bytes : Bytes) (r j : Nat) : BitVec 64 :=
  integerLogical (read64 bytes) input.labels.tensor r j
def seedWord (p : Parameters) (input : Inputs p) (bytes : Bytes) (r : Nat) : BitVec 64 :=
  integerLogical (read64 bytes) input.seeds.tensor r 0
def offsetWord (p : Parameters) (input : Inputs p) (bytes : Bytes) (r : Nat) : BitVec 64 :=
  integerLogical (read64 bytes) input.offsets.tensor r 0

/-- Only individual initial-memory loads are bound here. There is no execution,
recurrence, result or probability assumption in this storage interface. -/
structure Loaded {α : Type} [Inhabited α] (p : Parameters) (input : Inputs p)
    (bytes : Bytes) (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) : Prop where
  scores : ∀ a,a < input.scores.metadata.numel → (s.mem p.scores a).read .real=
    interpret (read32 bytes ((VeriTile.Torch.FP32.dataView input.scores input.scoreLaws.laws
      input.scoreLaws.contiguous).address a))
  labels : ∀ r j,r<p.batches → j<p.tokens →
    (s.mem p.indices (r*p.strideIndexBatch+j*p.strideIndexToken)).read .int=
      (read64 bytes (input.labels.tensor.metadata.dataPtr.toNat+
        8*(r*p.strideIndexBatch+j*p.strideIndexToken))).toInt
  seeds : ∀ r,r<p.batches → (s.mem p.seeds r).read .int=
    (read64 bytes (input.seeds.tensor.metadata.dataPtr.toNat+8*r)).toInt
  offsets : ∀ r,r<p.batches → (s.mem p.offsets r).read .int=
    (read64 bytes (input.offsets.tensor.metadata.dataPtr.toNat+8*r)).toInt
  undefinedSeeds : ∀ r,∃ word : BitVec 64,s.undef .int p.seeds r=word.toInt

variable {α : Type} [Inhabited α]

theorem score_read (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s)
    (r j : Nat) (hr : r<p.batches) (hj : j<p.tokens) :
    (s.mem p.scores (r*p.strideScoreBatch+j*p.strideScoreToken)).read .real=
      interpret (scoreWord p input bytes r j) :=
  source_score p s input.scores input.scoreLaws (read32 bytes) interpret loaded.scores r j hr hj

theorem integer_read {rows columns sb st : Nat} (v : IntegerView rows columns sb st)
    (bytes : Bytes) (r j : Nat) (hr : r<rows) (hj : j<columns) :
    read64 bytes (v.tensor.metadata.dataPtr.toNat+8*(r*sb+j*st))=
      integerLogical (read64 bytes) v.tensor r j := by
  have h := (integer_logical (read64 bytes) v.tensor v.laws v.ordinary r j
    (by simpa [v.rowCount] using hr) (by simpa [v.columnCount] using hj)).1
  simpa only [v.batchStride,v.tokenStride] using h

theorem label_read (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s)
    (r j : Nat) (hr : r<p.batches) (hj : j<p.tokens) :
    (s.mem p.indices (r*p.strideIndexBatch+j*p.strideIndexToken)).read .int=
      (labelWord p input bytes r j).toInt := by
  rw [loaded.labels r j hr hj,integer_read input.labels bytes r j hr hj]
  rfl

theorem seed_read (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s)
    (r : Nat) (hr : r<p.batches) :
    (s.mem p.seeds r).read .int=(seedWord p input bytes r).toInt := by
  have h := integer_read input.seeds bytes r 0 hr (by decide)
  simp only [Nat.mul_one,Nat.zero_mul,Nat.add_zero] at h
  rw [loaded.seeds r hr,h]
  rfl

theorem offset_read (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s)
    (r : Nat) (hr : r<p.batches) :
    (s.mem p.offsets r).read .int=(offsetWord p input bytes r).toInt := by
  have h := integer_read input.offsets bytes r 0 hr (by decide)
  simp only [Nat.mul_one,Nat.zero_mul,Nat.add_zero] at h
  rw [loaded.offsets r hr,h]
  rfl

theorem seed_domain (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (interpret : VeriTile.Torch.FP32.Word → α) (s : State α) (loaded : Loaded p input bytes interpret s) :
    SeedDomain p s := by
  apply KernelMemory.seed_domain p s
  · intro r hr
    exact ⟨seedWord p input bytes r,seed_read p input bytes interpret s loaded r hr⟩
  · exact loaded.undefinedSeeds

/-- Scalar readback of the exact FP32 ratio used by the unchanged Python
normalization guard, before interpreting it as a Real probability. -/
structure Probabilities (p : Parameters) (input : Inputs p) (bytes : Bytes) where
  finite : Fin p.batches → Fin p.tokens → Bool
  units : Fin p.batches → Fin p.tokens → Int
  checked : ∀ r,Contract.probabilityRow p.tokens (finite r) (units r)=Bool.true
  readback : ∀ r j,(scoreWord p input bytes r.val j.val).toRat?=
    some ((units r j : Rat)/(2:Rat)^149)

theorem probability_value (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (observed : Probabilities p input bytes) (r : Fin p.batches) (j : Fin p.tokens) :
    denote (scoreWord p input bytes r.val j.val)=(observed.units r j : ℝ)/(2:ℝ)^149 := by
  simp [denote,observed.readback,Rat.cast_div,Rat.cast_pow]

def weights (p : Parameters) (input : Inputs p) (bytes : Bytes) (r j : Nat) : ℝ :=
  if j<p.tokens then denote (scoreWord p input bytes r j) else 0

theorem probability_domain (p : Parameters) (input : Inputs p) (bytes : Bytes)
    (observed : Probabilities p input bytes) (positive : 0<p.tokens) (r : Nat) (hr : r<p.batches) :
    Spec.Domain p.tokens (weights p input bytes r) := by
  apply KernelMemory.probability_domain p.tokens positive (observed.finite ⟨r,hr⟩)
    (observed.units ⟨r,hr⟩) _ (observed.checked ⟨r,hr⟩)
  intro j
  simpa only [weights,j.isLt,↓reduceIte] using probability_value p input bytes observed ⟨r,hr⟩ j

#axiomsClean wordToReal_encode
#axiomsClean score_read
#axiomsClean label_read
#axiomsClean seed_read
#axiomsClean offset_read
#axiomsClean seed_domain
#axiomsClean probability_domain
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.TensorInputs
