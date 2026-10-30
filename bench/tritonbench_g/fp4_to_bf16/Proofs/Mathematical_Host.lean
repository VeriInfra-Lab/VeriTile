import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_KernelCorrect

/-! The original wrapper's packed-uint8, contiguous logical tensor profile.
The wrapper allocates a fresh bf16 output, doubles its final dimension, and
launches ceilDiv(numel, 512) programs. This file executes the actual admitted
kernel for each program; its expected output reads only the initial input.
Python allocation/shape lowering and native launch/concurrency remain separate
trust boundaries. In particular, an absent Python dtype assertion is not a
claim that arbitrary input storage has packed-uint8 semantics. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.Host
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- A nonscalar shape, exactly the domain on which x.shape[-1] is defined. -/
structure Shape where
  leading : List Nat
  last : Nat

def Shape.input (s : Shape) : List Nat := s.leading ++ [s.last]
def Shape.output (s : Shape) : List Nat := s.leading ++ [s.last * 2]
def Shape.count (s : Shape) : Nat := s.leading.prod * s.last

theorem input_numel (s : Shape) : s.input.prod = s.count := by
  simp [Shape.input, Shape.count]
theorem output_numel (s : Shape) : s.output.prod = s.count * 2 := by
  simp [Shape.output, Shape.count, Nat.mul_assoc]

structure Input where
  input : RegionName
  output : RegionName
  shape : Shape

def parameters (h : Input) : Parameters := ⟨h.input, h.output, h.shape.count, 512⟩
def grid (h : Input) : Nat := (h.shape.count + 511) / 512

/-- The actual storage domain of the packed input. Output memory is arbitrary
before the launch, matching torch.empty rather than a zero-initialization. -/
def PackedInput (h : Input) (initial : State Word) : Prop :=
  ∀ i, i < h.shape.count → ∃ b : BitVec 8, initial.mem h.input i = .mk .nat b.toNat

def nibble (h : Input) (initial : State Word) (j : Nat) : Bits.Nibble :=
  let byte := BitVec.ofNat 8 ((initial.mem h.input (j / 2)).read .nat)
  if j % 2 = 0 then Bits.high byte else Bits.low byte

def value (h : Input) (initial : State Word) (j : Nat) : Word :=
  embed ⟨Bits.expectedBits (nibble h initial j)⟩
def cell (h : Input) (initial : State Word) (j : Nat) : Cell Word :=
  .mk .bf16 (value h initial j)

/-- A fresh program-local register file; shared logical memory and arbitrary
masked-load undef values are preserved. -/
def programState (h : Input) (pid : Nat) (previous : State Word) : State Word :=
  { previous with
    regs := fun _ _ _ => none
    pids := fun a => if a = 0 then pid else 0
    numPids := fun a => if a = 0 then grid h else 1 }

/-- Operational launch prefix: only actual kernel execution advances memory. -/
def runPrefix (rest : Algebra Word) (h : Input) (initial : State Word) : Nat → Option (State Word)
  | 0 => some initial
  | k + 1 => (runPrefix rest h initial k).bind fun previous =>
      FP.Structural.exec (backend rest) (source (parameters h)) (programState h k previous)

def run (rest : Algebra Word) (h : Input) (initial : State Word) : Option (State Word) :=
  runPrefix rest h initial (grid h)

/-- Total witness supplied by the existing successful kernel execution proof;
this is not the independent output reference. -/
def prefixState (rest : Algebra Word) (h : Input) (initial : State Word) : Nat → State Word
  | 0 => initial
  | k + 1 => (execution rest (parameters h)
      (programState h k (prefixState rest h initial k))).val

theorem run_prefix (rest : Algebra Word) (h : Input) (initial : State Word) (k : Nat) :
    runPrefix rest h initial k = some (prefixState rest h initial k) := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [runPrefix, ih, Option.bind_some, prefixState]
    exact (execution rest (parameters h) (programState h k (prefixState rest h initial k))).property

private theorem block_expected (h : Input) (initial previous : State Word) (k : Nat)
    (same : previous.mem h.input = initial.mem h.input)
    (j : TileIndex [(parameters h).block * 2]) :
    expectedValue (parameters h) (programState h k previous) j =
      value h initial (k * 1024 + j.1.val) := by
  have hd : (k * 1024 + j.1.val) / 2 = k * 512 + j.1.val / 2 := by omega
  have hm : (k * 1024 + j.1.val) % 2 = j.1.val % 2 := by omega
  simp only [expectedValue, initialNibble, inputOffset, parameters, programState,
    value, nibble, hd, hm, same, ↓reduceIte]

/-- Exact prefix memory, expressed solely from the initial memory. -/
theorem prefix_memory (rest : Algebra Word) (h : Input) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial) (k : Nat) :
    ∀ r o, (prefixState rest h initial k).mem r o =
      if r = h.output ∧ o < h.shape.count * 2 ∧ o < k * 1024
      then cell h initial o else initial.mem r o := by
  induction k with
  | zero => intro r o; simp [prefixState]
  | succ k ih =>
    let previous := prefixState rest h initial k
    let start := programState h k previous
    have same : previous.mem h.input = initial.mem h.input := by
      funext o
      simpa only [fresh, false_and, ↓reduceIte] using ih h.input o
    have ht : ActiveBytes (parameters h) start := by
      intro i hi
      have hi' : k * 512 + i.val < h.shape.count := by
        simpa only [inputOffset, parameters, start, programState, ↓reduceIte] using hi
      obtain ⟨b, hb⟩ := typed (k * 512 + i.val) hi'
      refine ⟨b, ?_⟩
      change previous.mem h.input (k * 512 + i.val) = _
      rw [same, hb]
    have memory : (prefixState rest h initial (k + 1)).mem =
        (written (parameters h) start).mem := execution_memory rest (parameters h) start ht
    intro r o
    rw [memory]
    by_cases here : r = h.output ∧ k * 1024 ≤ o ∧ o < (k + 1) * 1024 ∧ o < h.shape.count * 2
    · obtain ⟨hr, hlo, hhi, hn⟩ := here
      subst r
      let j : TileIndex [(parameters h).block * 2] :=
        (⟨o - k * 1024, by dsimp [parameters]; omega⟩, PUnit.unit)
      have hoff : outputOffset (parameters h) start j = o := by
        dsimp [outputOffset, parameters, start, programState, j]
        omega
      have ha : active (parameters h) start j := by
        change outputOffset (parameters h) start j < h.shape.count * 2
        rw [hoff]
        exact hn
      have hv := readback (parameters h) start j ha
      rw [hoff] at hv
      change (written (parameters h) start).mem h.output o = _ at hv
      rw [hv]
      have he := block_expected h initial previous k same j
      have hj : k * 1024 + j.1.val = o := by dsimp [j]; omega
      rw [hj] at he
      change Cell.mk .bf16 (expectedValue (parameters h) start j) = _
      rw [he]
      simp only [hn, hhi, and_self, ↓reduceIte, cell]
    · have untouched : r ≠ (parameters h).output ∨
          ∀ j : TileIndex [(parameters h).block * 2], active (parameters h) start j →
            o ≠ outputOffset (parameters h) start j := by
        by_cases hr : r = h.output
        · right
          intro j ha he
          have hj := j.1.isLt
          dsimp [active, outputOffset, parameters, start, programState] at ha he hj
          apply here
          refine ⟨hr, ?_, ?_, ?_⟩ <;> omega
        · exact Or.inl hr
      rw [frame (parameters h) start r o untouched]
      change previous.mem r o = _
      rw [ih]
      congr 1
      apply propext
      constructor
      · rintro ⟨hr, hn, hk⟩
        exact ⟨hr, hn, by omega⟩
      · rintro ⟨hr, hn, hk⟩
        refine ⟨hr, hn, ?_⟩
        by_contra old
        apply here
        exact ⟨hr, by omega, hk, hn⟩

theorem grid_covers (h : Input) : h.shape.count * 2 ≤ grid h * 1024 := by
  unfold grid
  omega

/-- Independent final reference over initial packed bytes, with a full frame. -/
def expectedMemory (h : Input) (initial : State Word) : RegionName → Nat → Cell Word :=
  fun r o => if r = h.output ∧ o < h.shape.count * 2 then cell h initial o else initial.mem r o

theorem launch_memory (rest : Algebra Word) (h : Input) (initial : State Word)
    (fresh : h.input ≠ h.output) (typed : PackedInput h initial) :
    (prefixState rest h initial (grid h)).mem = expectedMemory h initial := by
  funext r o
  rw [prefix_memory rest h initial fresh typed]
  unfold expectedMemory
  have cover := grid_covers h
  by_cases active : r = h.output ∧ o < h.shape.count * 2
  · have inGrid : o < grid h * 1024 := by omega
    simp only [active.1, active.2, inGrid, and_self, ↓reduceIte]
  · have hnot : ¬ (r = h.output ∧ o < h.shape.count * 2 ∧ o < grid h * 1024) := by
      tauto
    simp only [active, hnot, ↓reduceIte]

theorem expected_decode (h : Input) (initial : State Word) (j : Nat) :
    (⟨Bits.expectedBits (nibble h initial j)⟩ : FP.Value .bf16).decode =
      .finite (decide (8 ≤ (nibble h initial j).toNat)) (Bits.magnitude (nibble h initial j)) := by
  have hb := (Bits.all_nibbles (nibble h initial j)).2.1
  have hd := (Bits.all_nibbles (nibble h initial j)).2.2
  have he : Bits.decoded (nibble h initial j) =
      (⟨Bits.expectedBits (nibble h initial j)⟩ : FP.Value .bf16) := by
    cases he : Bits.decoded (nibble h initial j)
    simp_all
  rw [he] at hd
  exact hd

#specNonCircular expectedMemory avoiding [run, runPrefix, prefixState, execution, FP.Structural.exec]
#axiomsClean input_numel
#axiomsClean output_numel
#axiomsClean run_prefix
#axiomsClean prefix_memory
#axiomsClean launch_memory
#axiomsClean expected_decode
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.Host
