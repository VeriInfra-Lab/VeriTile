import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.AdmissionCore
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphTensor

/-! Executable finite-index/control semantics for the repaired source. Loads
use signed addresses, not a silently truncated negative-to-Nat projection.
The theorem proves that every active signed address is the corresponding
nonnegative source address. The finite signed loop executes its own test and
increment; its result is then composed with the actual typed source executor.
This is a source-semantics refinement, not a compiler implementation proof. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.FiniteExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open Whole Index64
noncomputable section
local instance : Inhabited SourceGraph.Expr := ⟨.zero⟩
set_option maxHeartbeats 2500000
set_option linter.unusedSimpArgs false

abbrev Word := SourceGraph.Expr
abbrev Memory := SourceGraph.Memory
abbrev Lane (p : Parameters) := TileIndex [p.BM,p.BN]
abbrev Carry (p : Parameters) := Lane p → Word

def cast32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64

def row (p : Parameters) (s : State Word) (i : Fin p.BM) : BitVec 64 :=
  cast32 (s.pids 0) * word p.BM + cast32 i.val

def col (p : Parameters) (s : State Word) (j : Fin p.BN) : BitVec 64 :=
  cast32 (s.pids 1) * word p.BN + cast32 j.val

def kOffset (p : Parameters) (c : BitVec 64) (e : Fin p.BK) : BitVec 64 :=
  c * word p.BK + cast32 e.val

def aOffset (p : Parameters) (s : State Word) (c : BitVec 64)
    (i : Fin p.BM) (e : Fin p.BK) : BitVec 64 :=
  row p s i * runtimeDimension p.K + kOffset p c e

def bOffset (p : Parameters) (s : State Word) (c : BitVec 64)
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) : BitVec 64 :=
  row p s i * runtimeDimension p.N * runtimeDimension p.K +
    col p s j * runtimeDimension p.K + kOffset p c e

def outOffset (p : Parameters) (s : State Word) (i : Lane p) : BitVec 64 :=
  row p s i.1 * runtimeDimension p.N + col p s i.2.1

def bound (p : Parameters) : BitVec 64 :=
  (runtimeDimension p.K).sdiv (word p.BK)

def ProgramIds (p : Parameters) (s : State Word) : Prop :=
  s.pids 0 < ValueProofs.gridM p ∧ s.pids 1 < ValueProofs.gridN p

/-- Byte-offset formation is finite64 too, before projection to a stored
FP32 cell. Physical base addition is covered by input/output_byte_addresses. -/
def byteOffset (offset : BitVec 64) : BitVec 64 := offset * word 4

/-- Signed byte memory extension. Negative or unaligned addresses do not
alias Nat cell zero. Every executed address is proved a nonnegative aligned
four-byte cell inside its allocation before this projection is simplified. -/
def signedByteMemory (mem : Memory) (r : RegionName) (bytes : Int) : Cell Word :=
  if 0 ≤ bytes ∧ bytes % 4 = 0 then mem r (bytes.toNat / 4) else .mk .real .zero

@[simp] theorem signedByteMemory_nat (mem : Memory) (r : RegionName) (o : Nat) :
    signedByteMemory mem r ((o*4 : Nat) : Int) = mem r o := by
  unfold signedByteMemory
  rw [if_pos (by omega),Int.toNat_natCast]
  simp

theorem expressions (p : Parameters) (s : State Word) (h : IntegerAgreement p)
    (hp : ProgramIds p s) (c : Nat) (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    aOffset p s (word c) i e = Index64.aOffset p (s.pids 0) c i.val e.val ∧
    bOffset p s (word c) i j e = Index64.bOffset p (s.pids 0) (s.pids 1) c i.val j.val e.val := by
  obtain ⟨hm,hn⟩ := h.programIds ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩
  simp only [aOffset,bOffset,row,col,kOffset,cast32,runtime_dimension_word,
    hm,hn,h.laneM i,h.laneN j,h.laneK e,Index64.aOffset,Index64.bOffset,coordinate]
  exact ⟨trivial,trivial⟩

theorem output_expression (p : Parameters) (s : State Word) (h : IntegerAgreement p)
    (hp : ProgramIds p s) (i : Lane p) :
    outOffset p s i = Index64.outOffset p (s.pids 0) (s.pids 1) i.1.val i.2.1.val := by
  obtain ⟨hm,hn⟩ := h.programIds ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩
  simp only [outOffset,row,col,cast32,runtime_dimension_word,hm,hn,
    h.laneM i.1,h.laneN i.2.1,Index64.outOffset,coordinate]

theorem input_signed_addresses (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    (aOffset p s (word c.val) i e).toInt =
      (SourceGraph.aOffset p s i.val c.val e.val : Int) ∧
    (bOffset p s (word c.val) i j e).toInt =
      (SourceGraph.bOffset p s i.val j.val c.val e.val : Int) := by
  obtain ⟨ha,hb⟩ := expressions p s h hp c.val i j e
  rw [ha,hb]
  exact h.addresses.inputs ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ c i j e

theorem output_signed_address (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (i : Lane p) :
    (outOffset p s i).toInt = (SourceGraph.outputOffset p s i : Int) := by
  rw [output_expression p s h hp i]
  exact h.addresses.output ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ i.1 i.2.1

/-- The actual finite offset is used in both signed memory and physical byte
address formation. This theorem proves the complete four-byte access window. -/
theorem input_byte_addresses (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) (baseA baseB : Nat)
    (wa : baseA+p.M*p.K*4 ≤ 2^64) (wb : baseB+p.M*p.N*p.K*4 ≤ 2^64) :
    (bytePointer baseA (aOffset p s (word c.val) i e)).toNat =
      baseA+SourceGraph.aOffset p s i.val c.val e.val*4 ∧
    baseA ≤ (bytePointer baseA (aOffset p s (word c.val) i e)).toNat ∧
    (bytePointer baseA (aOffset p s (word c.val) i e)).toNat+3 < baseA+p.M*p.K*4 ∧
    (bytePointer baseB (bOffset p s (word c.val) i j e)).toNat =
      baseB+SourceGraph.bOffset p s i.val j.val c.val e.val*4 ∧
    baseB ≤ (bytePointer baseB (bOffset p s (word c.val) i j e)).toNat ∧
    (bytePointer baseB (bOffset p s (word c.val) i j e)).toNat+3 < baseB+p.M*p.N*p.K*4 := by
  obtain ⟨ha,hb⟩ := expressions p s h hp c.val i j e
  rw [ha,hb,Index64.aOffset_eq,Index64.bOffset_eq]
  obtain ⟨ba,bb⟩ := h.inputBytes ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ c i j e
  change SourceGraph.aOffset p s i.val c.val e.val*4+3 < p.M*p.K*4 at ba
  change SourceGraph.bOffset p s i.val j.val c.val e.val*4+3 < p.M*p.N*p.K*4 at bb
  obtain ⟨aa,ab,ac⟩ := h.pointerWindow baseA (p.M*p.K)
    (SourceGraph.aOffset p s i.val c.val e.val) wa (by omega)
  obtain ⟨ba,bb,bc⟩ := h.pointerWindow baseB (p.M*p.N*p.K)
    (SourceGraph.bOffset p s i.val j.val c.val e.val) wb (by omega)
  exact ⟨aa,ab,ac,ba,bb,bc⟩

theorem output_byte_address (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (i : Lane p) (base : Nat)
    (window : base+p.M*p.N*4 ≤ 2^64) :
    (bytePointer base (outOffset p s i)).toNat = base+SourceGraph.outputOffset p s i*4 ∧
    base ≤ (bytePointer base (outOffset p s i)).toNat ∧
    (bytePointer base (outOffset p s i)).toNat+3 < base+p.M*p.N*4 := by
  rw [output_expression p s h hp i,Index64.outOffset_eq]
  have hb := h.outputBytes ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ i.1 i.2.1
  change SourceGraph.outputOffset p s i*4+3 < p.M*p.N*4 at hb
  exact h.pointerWindow base (p.M*p.N) (SourceGraph.outputOffset p s i) window (by omega)

/-- Exact signed byte offsets used by the load interpreter. The bounds are
on every byte, not only the mathematical element number. -/
theorem input_signed_bytes (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    (byteOffset (aOffset p s (word c.val) i e)).toInt =
      ((SourceGraph.aOffset p s i.val c.val e.val*4 : Nat) : Int) ∧
    (byteOffset (bOffset p s (word c.val) i j e)).toInt =
      ((SourceGraph.bOffset p s i.val j.val c.val e.val*4 : Nat) : Int) := by
  obtain ⟨ha,hb⟩ := expressions p s h hp c.val i j e
  obtain ⟨ba,bb⟩ := h.inputBytes ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ c i j e
  change SourceGraph.aOffset p s i.val c.val e.val*4+3 < p.M*p.K*4 at ba
  change SourceGraph.bOffset p s i.val j.val c.val e.val*4+3 < p.M*p.N*p.K*4 at bb
  rw [ha,hb,Index64.aOffset_eq,Index64.bOffset_eq]
  change (word (SourceGraph.aOffset p s i.val c.val e.val) * word 4).toInt = _ ∧
    (word (SourceGraph.bOffset p s i.val j.val c.val e.val) * word 4).toInt = _
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_mul]
  exact ⟨word_toInt (by have := h.storage.A; omega),
    word_toInt (by have := h.storage.B; omega)⟩

theorem output_signed_bytes (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (i : Lane p) :
    (byteOffset (outOffset p s i)).toInt =
      ((SourceGraph.outputOffset p s i*4 : Nat) : Int) := by
  have hb := h.outputBytes ⟨s.pids 0,hp.1⟩ ⟨s.pids 1,hp.2⟩ i.1 i.2.1
  change SourceGraph.outputOffset p s i*4+3 < p.M*p.N*4 at hb
  rw [output_expression p s h hp i,Index64.outOffset_eq]
  change (word (SourceGraph.outputOffset p s i)*word 4).toInt = _
  rw [← BitVec.ofNat_mul]
  exact word_toInt (by have := h.storage.output; omega)

def blockSum (plans : SourceGraph.PlanSelector) (p : Parameters)
    (s : State Word) (c : BitVec 64) (i : Lane p) : Word :=
  let args := fun e : Fin p.BK => SourceGraph.Expr.mul
    ((signedByteMemory s.mem p.A (byteOffset (aOffset p s c i.1 e)).toInt).read .real)
    ((signedByteMemory s.mem p.B (byteOffset (bOffset p s c i.1 i.2.1 e)).toInt).read .real)
  SourceGraph.treeValue args (plans p.BK args).tree

theorem block_sum_eq (plans : SourceGraph.PlanSelector) (p : Parameters)
    (s : State Word) (h : IntegerAgreement p) (hp : ProgramIds p s)
    (c : Nat) (hc : c < p.K/p.BK) (i : Lane p) :
    blockSum plans p s (word c) i = SourceGraph.blockSum plans p s c i := by
  have ha := fun e => (input_signed_bytes p s h hp ⟨c,hc⟩ i.1 i.2.1 e).1
  have hb := fun e => (input_signed_bytes p s h hp ⟨c,hc⟩ i.1 i.2.1 e).2
  simp only [blockSum,ha,hb,signedByteMemory_nat,SourceGraph.blockSum]

/-- The signed test occurs even at zero fuel: a valid stopping value succeeds,
whereas an exhausted still-active execution fails. Fuel does not silently
replace the source loop condition. -/
def loop (plans : SourceGraph.PlanSelector) (p : Parameters) (s : State Word) :
    Nat → BitVec 64 → Carry p → Option (Carry p)
  | 0, c, carry => if c.toInt < (bound p).toInt then none else some carry
  | fuel+1, c, carry => if c.toInt < (bound p).toInt then
      loop plans p s fuel (c+word 1) (fun i => .add (carry i) (blockSum plans p s c i))
    else some carry

theorem loop_eq (plans : SourceGraph.PlanSelector) (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) (fuel c : Nat)
    (sum : c+fuel = p.K/p.BK) :
    loop plans p s fuel (word c) (SourceGraph.accumulated plans p s c) =
      some (SourceGraph.accumulated plans p s (p.K/p.BK)) := by
  induction fuel generalizing c with
  | zero =>
    have hc : c = p.K/p.BK := by omega
    have ht : ¬ (word c).toInt < (bound p).toInt := by
      simpa only [bound,runtime_dimension_word] using
        (h.loopTest c (by omega)).not.mpr (by omega)
    rw [loop,if_neg ht,hc]
  | succ fuel ih =>
    have hc : c < p.K/p.BK := by omega
    have ht : (word c).toInt < (bound p).toInt := by
      simpa only [bound,runtime_dimension_word] using (h.loopTest c (by omega)).mpr hc
    have hs : word c+word 1 = word (c+1) := (BitVec.ofNat_add c 1).symm
    have hb : (fun i => SourceGraph.Expr.add (SourceGraph.accumulated plans p s c i)
        (blockSum plans p s (word c) i)) = SourceGraph.accumulated plans p s (c+1) := by
      funext i
      rw [block_sum_eq plans p s h hp c hc i]
      rfl
    rw [loop,if_pos ht,hs,hb]
    exact ih (c+1) (by omega)

/-- Stores compare the signed finite address with an observable Nat cell;
no negative address is reinterpreted as Nat zero. -/
def program (plans : SourceGraph.PlanSelector) (p : Parameters) (s : State Word) : Option Memory :=
  (loop plans p s (bound p).toNat (word 0) (fun _ => .zero)).map fun carry =>
    (TileShape.allIndices [p.BM,p.BN]).foldl (fun mem i =>
      fun r o => if r = p.output ∧ ((o*4 : Nat) : Int) = (byteOffset (outOffset p s i)).toInt then
        .mk .real (carry i) else mem r o) s.mem

theorem program_eq (plans : SourceGraph.PlanSelector) (p : Parameters) (s : State Word)
    (h : IntegerAgreement p) (hp : ProgramIds p s) :
    program plans p s = some (SourceGraph.expectedMemory plans p s) := by
  have hb : (bound p).toNat = p.K/p.BK := by simpa [bound] using h.loopBound
  have hl := loop_eq plans p s h hp (p.K/p.BK) 0 (by omega)
  change loop plans p s (p.K/p.BK) (word 0) (fun _ => .zero) = _ at hl
  simp only [program,hb,hl,Option.map_some,SourceGraph.expectedMemory]
  congr 1
  congr 1
  funext mem i r o
  rw [output_signed_bytes p s h hp i]
  have he : ((o*4 : Nat) : Int) = ((SourceGraph.outputOffset p s i*4 : Nat) : Int) ↔
      o = SourceGraph.outputOffset p s i := by omega
  simp only [he]

/-- Direct bridge to execution of every typed source statement, including
its prologue, load/reduction/carry loop and unmasked store. -/
theorem program_source_execution (rest : Algebra Word) (plans : SourceGraph.PlanSelector)
    (p : Parameters) (s : State Word) (h : IntegerAgreement p) (hp : ProgramIds p s) :
    program plans p s =
      (FP.Structural.exec (SourceGraph.engine rest plans) (SourceGraph.source p) s).map State.mem := by
  obtain ⟨final,he,hm,_⟩ := SourceGraph.complete_correct rest plans p s
  rw [program_eq plans p s h hp,he,Option.map_some,hm]

def gridPrefix (plans : SourceGraph.PlanSelector) (p : Parameters) (initial : State Word) :
    Nat → Option Memory
  | 0 => some initial.mem
  | k+1 => (gridPrefix plans p initial k).bind fun mem =>
      program plans p (SourceGraph.programState p initial mem k)

def grid (plans : SourceGraph.PlanSelector) (p : Parameters) (initial : State Word) : Option Memory :=
  gridPrefix plans p initial (SourceGraph.gridCount p)

theorem grid_prefix_eq (plans : SourceGraph.PlanSelector) (p : Parameters)
    (initial : State Word) (h : IntegerAgreement p) (k : Nat) (hk : k ≤ SourceGraph.gridCount p) :
    gridPrefix plans p initial k = some (SourceGraph.referenceGridPrefix plans p initial k) := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [gridPrefix,ih (by omega),Option.bind_some]
    have hb := SourceGraph.coordinate_bounds p k (by omega)
    exact program_eq plans p _ h (by
      simpa [ProgramIds,SourceGraph.programState,SourceGraph.gridM,SourceGraph.gridN,
        ValueProofs.gridM,ValueProofs.gridN] using hb)

theorem grid_source_execution (rest : Algebra Word) (plans : SourceGraph.PlanSelector)
    (p : Parameters) (initial : State Word) (h : IntegerAgreement p) :
    grid plans p initial = (SourceGraph.runGrid rest plans p initial).map State.mem := by
  obtain ⟨final,he,hm,_⟩ := SourceGraph.grid_correct rest plans p initial
  rw [grid,grid_prefix_eq plans p initial h _ (by rfl),he,Option.map_some,hm]
  rfl

#axiomsClean expressions
#axiomsClean input_signed_addresses
#axiomsClean output_signed_address
#axiomsClean input_byte_addresses
#axiomsClean output_byte_address
#axiomsClean input_signed_bytes
#axiomsClean output_signed_bytes
#axiomsClean loop_eq
#axiomsClean program_source_execution
#axiomsClean grid_source_execution
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.FiniteExecution
