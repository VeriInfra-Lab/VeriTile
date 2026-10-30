import bench.tritonbench_g.iv_dependent_matmul.Proofs.Safety

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Index64
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section
set_option maxHeartbeats 3000000

/-- Ordinary allocated-buffer addressability. Unlike the old IndexDomain, this
has no i32/i64 envelope condition on inactive lanes or speculative pointers. -/
structure Addressable (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  input : a.baseA+4*(r.rows*r.inner) ≤ 2^64
  weight : a.baseB+4*(r.inner*r.columns) ≤ 2^64
  output : a.baseC+4*(r.rows*r.columns) ≤ 2^64

/-- Native compilation/launch admissibility, not an extra array-size restriction.
Triton arange widths are at most 2^20; the looser 2^31 bounds suffice below.
The CUDA x-grid limit is 2^31-1. Setup failure remains an ordinary API outcome. -/
structure Launchable (r : Request) : Prop where
  rowsBlock : r.blockRows ≤ 2^31
  columnsBlock : r.blockColumns ≤ 2^31
  innerBlock : r.blockInner ≤ 2^31
  grid : programs r < 2^31

def word64 (n : Nat) := n % 2^64

def signed64 (n : Nat) : Int :=
  if word64 n < 2^63 then (word64 n : Int) else (word64 n : Int)-(2^64 : Int)

/-- i64 modular pointer-stage arithmetic. Out-of-range inactive/staged offsets
may wrap. The active final address theorem does not assume they never wrap. -/
def staged64 (base k width stride : Nat) :=
  word64 (word64 base + word64 (word64 (k*width)*stride))

theorem stage_residue (base k width stride : Nat) :
    staged64 base k width stride = word64 (base+k*width*stride) := by
  simp [staged64,word64,Nat.add_mod,Nat.mul_mod]

theorem i64_small (n : Nat) (h : n<2^63) : signed64 n=(n : Int) := by
  have h64 : n<2^64 := by omega
  simp only [signed64,word64,Nat.mod_eq_of_lt h64,if_pos h]

/-- Element offsets from any real FP32 object fit signed i64. -/
theorem allocated_offset (base extent o : Nat) (h : base+4*extent≤2^64) (ho : o<extent) :
    o<2^63 ∧ base+4*o<2^64 ∧ signed64 o=(o : Int) := by
  have hs : o<2^63 := by omega
  exact ⟨hs,by omega,i64_small o hs⟩

/-- Pointer extension after modular element arithmetic recovers the actual
64-bit byte address for every active allocated element. -/
theorem allocated_byte (base extent o : Nat) (h : base+4*extent≤2^64) (ho : o<extent) :
    word64 (base+4*word64 o)=base+4*o := by
  have hs := allocated_offset base extent o h ho
  have h64 : o<2^64 := by omega
  simp only [word64,Nat.mod_eq_of_lt h64,Nat.mod_eq_of_lt hs.2.1]

theorem a_staging (r : Request) (s : State Word) (k : Nat)
    (i : TileIndex [r.blockRows,r.blockInner]) :
    staged64 (aBase r.parameters s i) k r.blockInner 1 = word64 (aOffset r.parameters s k i) := by
  rw [stage_residue]
  rfl

theorem b_staging (r : Request) (s : State Word) (k : Nat)
    (i : TileIndex [r.blockInner,r.blockColumns]) :
    staged64 (bBase r.parameters s i) k r.blockInner (max r.columns 1) =
      word64 (bOffset r.parameters s k i) := by
  rw [stage_residue]
  rfl

/-- Fits all control arithmetic before any element-stride multiplication.
The bounds are derived from allocated extents, including for K=0; no old
signed-i32 address restriction is reintroduced. -/
theorem shape_control (r : Request) (s : State Word) (a : Allocation r s)
    (d : Addressable r s a) (l : Launchable r) (pid : Nat) (hp : pid<programs r) :
    r.rows+r.blockRows<2^63 ∧ r.columns+r.blockColumns<2^63 ∧
    r.inner+4*r.blockInner+3<2^63 := by
  obtain ⟨hm,hn⟩ := nonempty_program r pid hp
  have hc := d.output
  have ha := d.input
  have hr : r.rows≤r.rows*r.columns := by nlinarith
  have hn' : r.columns≤r.rows*r.columns := by nlinarith
  have hk : r.inner≤r.rows*r.inner := by nlinarith
  have br := l.rowsBlock
  have bn := l.columnsBlock
  have bk := l.innerBlock
  omega

/-- Source K-mask subtraction, ceil-K count and all k+1/k+2/k+3 indices
fit signed i64. Staged element-stride products use stage_residue instead. -/
theorem loop_control (r : Request) (s : State Word) (a : Allocation r s)
    (d : Addressable r s a) (l : Launchable r) (pid k : Nat)
    (hp : pid<programs r) (hk : k<count r.parameters) :
    k*r.blockInner<r.inner ∧ (r.inner : Int)-(k*r.blockInner : Nat)=(r.inner-k*r.blockInner : Nat) ∧
    k+3<2^63 ∧ (k+3)*r.blockInner<2^63 := by
  have hs := shape_control r s a d l pid hp
  have hi := source_subtraction r k hk
  have hb := r.positiveInnerBlock
  have hle : k≤k*r.blockInner := by nlinarith
  refine ⟨hi.1,hi.2,?_,?_⟩
  · omega
  · nlinarith [hs.2.2]

/-- Both PID quotient/remainder coordinates fit before modulo and output masks. -/
theorem lane_coordinate (pid block lane : Nat) (hp : pid<2^31)
    (hb : block≤2^31) (hl : lane<block) : pid*block+lane<2^63 := by
  have hm := Nat.mul_le_mul (Nat.succ_le_of_lt hp) hb
  nlinarith

theorem program_coordinates (r : Request) (s : State Word) (l : Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    (∀ i : Fin r.blockRows, row r.parameters (instanceState pid s) i<2^63) ∧
    (∀ j : Fin r.blockColumns, col r.parameters (instanceState pid s) j<2^63) := by
  have hpid : pid<2^31 := lt_trans hp l.grid
  constructor
  · intro i
    exact lane_coordinate _ _ _ (lt_of_le_of_lt (Nat.div_le_self _ _) hpid) l.rowsBlock i.isLt
  · intro j
    exact lane_coordinate _ _ _ (lt_of_le_of_lt (Nat.mod_le _ _) hpid) l.columnsBlock j.isLt

theorem counts_control (r : Request) (s : State Word) (a : Allocation r s)
    (d : Addressable r s a) (l : Launchable r) (pid : Nat) (hp : pid<programs r) :
    tileRows r<2^63 ∧ tileColumns r<2^63 ∧ count r.parameters+3<2^63 := by
  have hs := shape_control r s a d l pid hp
  have hm := Nat.div_le_self (r.rows+r.blockRows-1) r.blockRows
  have hn := Nat.div_le_self (r.columns+r.blockColumns-1) r.blockColumns
  have hk := Nat.div_le_self (r.inner+r.blockInner-1) r.blockInner
  change (r.rows+r.blockRows-1)/r.blockRows<2^63 ∧
    (r.columns+r.blockColumns-1)/r.blockColumns<2^63 ∧
    (r.inner+r.blockInner-1)/r.blockInner+3<2^63
  omega

structure ActiveSafety (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  subtraction : ∀ k, k<count r.parameters → k*r.blockInner<r.inner ∧
    (r.inner : Int)-(k*r.blockInner : Nat)=(r.inner-k*r.blockInner : Nat)
  loadA : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockRows,r.blockInner],
    i.2.1.val<r.inner-k*r.blockInner →
      aOffset r.parameters (instanceState pid s) k i<r.rows*r.inner ∧
      signed64 (aOffset r.parameters (instanceState pid s) k i)=
        (aOffset r.parameters (instanceState pid s) k i : Int) ∧
      word64 (a.baseA+4*word64 (aOffset r.parameters (instanceState pid s) k i))=
        a.baseA+4*aOffset r.parameters (instanceState pid s) k i
  loadB : ∀ pid k, pid<programs r → ∀ i : TileIndex [r.blockInner,r.blockColumns],
    i.1.val<r.inner-k*r.blockInner →
      bOffset r.parameters (instanceState pid s) k i<r.inner*r.columns ∧
      signed64 (bOffset r.parameters (instanceState pid s) k i)=
        (bOffset r.parameters (instanceState pid s) k i : Int) ∧
      word64 (a.baseB+4*word64 (bOffset r.parameters (instanceState pid s) k i))=
        a.baseB+4*bOffset r.parameters (instanceState pid s) k i
  store : ∀ pid i, active r.parameters (instanceState pid s) i=Bool.true →
      outputOffset r.parameters (instanceState pid s) i<r.rows*r.columns ∧
      signed64 (outputOffset r.parameters (instanceState pid s) i)=
        (outputOffset r.parameters (instanceState pid s) i : Int) ∧
      word64 (a.baseC+4*word64 (outputOffset r.parameters (instanceState pid s) i))=
        a.baseC+4*outputOffset r.parameters (instanceState pid s) i

theorem active_safety (r : Request) (s : State Word) (a : Allocation r s)
    (d : Addressable r s a) : ActiveSafety r s a where
  subtraction := source_subtraction r
  loadA := by
    intro pid k hp i hm
    have h := loadA_bound r s pid k hp i hm
    exact ⟨h,(allocated_offset _ _ _ d.input h).2.2,allocated_byte _ _ _ d.input h⟩
  loadB := by
    intro pid k hp i hm
    have h := loadB_bound r s pid k hp i hm
    exact ⟨h,(allocated_offset _ _ _ d.weight h).2.2,allocated_byte _ _ _ d.weight h⟩
  store := by
    intro pid i hi
    have h := store_bound r s pid i hi
    exact ⟨h,(allocated_offset _ _ _ d.output h).2.2,allocated_byte _ _ _ d.output h⟩

/-- Public whole-source fixed-width facts. Allocation bounds are ordinary
physical buffer facts; control bounds follow from those plus native launch
limits. No numerical condition is used for index, mask or pointer laws. -/
structure MachineSafety (r : Request) (s : State Word) (a : Allocation r s) : Prop where
  accesses : ActiveSafety r s a
  shape : ∀ pid, pid<programs r → r.rows+r.blockRows<2^63 ∧
    r.columns+r.blockColumns<2^63 ∧ r.inner+4*r.blockInner+3<2^63
  coordinates : ∀ pid, pid<programs r →
    (∀ i : Fin r.blockRows, row r.parameters (instanceState pid s) i<2^63) ∧
    (∀ j : Fin r.blockColumns, col r.parameters (instanceState pid s) j<2^63)
  counts : ∀ pid, pid<programs r →
    tileRows r<2^63 ∧ tileColumns r<2^63 ∧ count r.parameters+3<2^63
  loop : ∀ pid k, pid<programs r → k<count r.parameters →
    k*r.blockInner<r.inner ∧ (r.inner : Int)-(k*r.blockInner : Nat)=(r.inner-k*r.blockInner : Nat) ∧
    k+3<2^63 ∧ (k+3)*r.blockInner<2^63
  stageA : ∀ pid k (i : TileIndex [r.blockRows,r.blockInner]),
    staged64 (aBase r.parameters (instanceState pid s) i) k r.blockInner 1 =
      word64 (aOffset r.parameters (instanceState pid s) k i)
  stageB : ∀ pid k (i : TileIndex [r.blockInner,r.blockColumns]),
    staged64 (bBase r.parameters (instanceState pid s) i) k r.blockInner (max r.columns 1) =
      word64 (bOffset r.parameters (instanceState pid s) k i)

theorem machine_safety (r : Request) (s : State Word) (a : Allocation r s)
    (d : Addressable r s a) (l : Launchable r) : MachineSafety r s a where
  accesses := active_safety r s a d
  shape := shape_control r s a d l
  coordinates := program_coordinates r s l
  counts := counts_control r s a d l
  loop := loop_control r s a d l
  stageA := fun pid => a_staging r (instanceState pid s)
  stageB := fun pid => b_staging r (instanceState pid s)

/-- The previously failing active output has a positive exact signed-i64 offset. -/
theorem repaired_boundary : signed64 (65536*32768)=(2147483648 : Int) ∧
    word64 (4*(65536*32768))=8589934592 := by decide

#axiomsClean machine_safety
#axiomsClean active_safety
#axiomsClean stage_residue
#axiomsClean repaired_boundary
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Index64
