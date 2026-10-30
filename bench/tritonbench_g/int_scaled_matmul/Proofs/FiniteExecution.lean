import bench.tritonbench_g.int_scaled_matmul.Proofs.FiniteIndex
import bench.tritonbench_g.int_scaled_matmul.Proofs.TotalScaled
import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordComplete

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteExecution
open VeriTile.Triton FP.Structural
open FiniteIndex
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
abbrev Word := ScaledWord.Word
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Launch where
  (M N K : Nat)
  config : Contract.Config
  (a b c : RegionName)

def blockParameters (x : Launch) : BlockWord.Parameters :=
  ⟨x.a,x.b,x.c,x.M,x.N,x.K,x.K,1,x.N,1,x.N,1,
    x.config.blockM,x.config.blockN,x.config.blockK,x.config.groupM⟩
def scaledParameters (x : Launch) (scale : RegionName) : ScaledWord.Parameters :=
  ⟨x.a,x.b,x.c,scale,x.M,x.N,x.K,x.K,1,x.N,1,x.N,1,1,1,
    x.config.blockM,x.config.blockN,x.config.blockK,x.config.groupM,
    Contract.ceil x.K x.config.blockK,decide (x.K%x.config.blockK=0)⟩
abbrev AIndex (x : Launch) := TileIndex [x.config.blockM,x.config.blockK]
abbrev BIndex (x : Launch) := TileIndex [x.config.blockK,x.config.blockN]
abbrev CIndex (x : Launch) := TileIndex [x.config.blockM,x.config.blockN]
def even (x : Launch) : Bool := decide (x.K%x.config.blockK=0)
def steps (x : Launch) := Contract.ceil x.K x.config.blockK

def naturalA (x : Launch) (wrapped : Bool) (s : State Word) (k : Nat) (i : AIndex x) : Int :=
  let r := naturalRow x.M x.N x.config (s.pids 0) i.1.val
  let kr := k*x.config.blockK+i.2.1.val
  BlockWord.wrap 8 (if (wrapped=Bool.true ∨ r<x.M) ∧
      ((wrapped=Bool.true ∧ even x=Bool.true) ∨ kr<x.K) then
    (s.mem x.a (selectedNaturalRow x.M x.N x.config wrapped (s.pids 0) i.1.val*x.K+kr)).read .int else 0)
def naturalB (x : Launch) (wrapped : Bool) (s : State Word) (k : Nat) (i : BIndex x) : Int :=
  let c := naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val
  let kr := k*x.config.blockK+i.1.val
  BlockWord.wrap 8 (if ((wrapped=Bool.true ∧ even x=Bool.true) ∨ kr<x.K) ∧
      (wrapped=Bool.true ∨ c<x.N) then
    (s.mem x.b (kr*x.N+selectedNaturalColumn x.M x.N x.config wrapped (s.pids 0) i.2.1.val)).read .int else 0)

def loadA (x : Launch) (wrapped : Bool) (s : State Word) (k : BitVec 32) (i : AIndex x) : Int :=
  BlockWord.wrap 8 (if (wrapped || (row x.M x.N x.config (s.pids 0) i.1.val).slt (word x.M)) &&
      ((wrapped && even x) || (word i.2.1.val).slt (word x.K-k*word x.config.blockK)) then
    (s.mem x.a (aOffset x.M x.N x.K x.config wrapped (s.pids 0) i.1.val i.2.1.val k.toNat).toNat).read .int else 0)
def loadB (x : Launch) (wrapped : Bool) (s : State Word) (k : BitVec 32) (i : BIndex x) : Int :=
  BlockWord.wrap 8 (if ((wrapped && even x) || (word i.1.val).slt (word x.K-k*word x.config.blockK)) &&
      (wrapped || (column x.M x.N x.config (s.pids 0) i.2.1.val).slt (word x.N)) then
    (s.mem x.b (bOffset x.M x.N x.config wrapped (s.pids 0) i.2.1.val i.1.val k.toNat).toNat).read .int else 0)

theorem row_mask (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (pid i : Nat)
    (hp : pid < count x.M x.N x.config) (hi : i < x.config.blockM) :
    (row x.M x.N x.config pid i).slt (word x.M)=Bool.true ↔ naturalRow x.M x.N x.config pid i<x.M := by
  have h := coordinates x.M x.N x.K x.config legal env pid i 0 hp hi (Contract.positive_tiles x.config legal).2.1
  have ms := (ceil_small x.M x.config.blockM (Contract.positive_tiles x.config legal).1
    (Contract.envelope_fields x.M x.N x.K x.config env).1).1
  rw [BitVec.slt_iff_toInt_lt,h.2.2.1,toInt ms]
  omega

theorem column_mask (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (pid j : Nat)
    (hp : pid < count x.M x.N x.config) (hj : j < x.config.blockN) :
    (column x.M x.N x.config pid j).slt (word x.N)=Bool.true ↔ naturalColumn x.M x.N x.config pid j<x.N := by
  have h := coordinates x.M x.N x.K x.config legal env pid 0 j hp (Contract.positive_tiles x.config legal).1 hj
  have ns := (ceil_small x.N x.config.blockN (Contract.positive_tiles x.config legal).2.1
    (Contract.envelope_fields x.M x.N x.K x.config env).2.1).1
  rw [BitVec.slt_iff_toInt_lt,h.2.2.2,toInt ns]
  omega

theorem loads (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (wrapped : Bool)
    (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) (k : Nat) (hk : k < steps x) :
    loadA x wrapped s (word k)=naturalA x wrapped s k ∧
    loadB x wrapped s (word k)=naturalB x wrapped s k := by
  have ks := ceil_small x.K x.config.blockK (Contract.positive_tiles x.config legal).2.2.1
    (Contract.envelope_fields x.M x.N x.K x.config env).2.2.1
  have small : k<2^32 := by change k<Contract.ceil x.K x.config.blockK at hk; have := ks.2.2.1; omega
  constructor
  · funext i
    have mask := inner_mask x.M x.N x.K x.config legal env k i.2.1.val (by exact Nat.le_of_lt hk) i.2.1.isLt
    simp only [remaining] at mask
    have address := (offset_words x.M x.N x.K x.config legal env wrapped (s.pids 0) i.1.val 0 i.2.1.val k hp i.1.isLt
      (Contract.positive_tiles x.config legal).2.1).1
    have bound := (offset_bounds x.M x.N x.K x.config legal env wrapped (s.pids 0) i.1.val 0 i.2.1.val k hp i.1.isLt
      (Contract.positive_tiles x.config legal).2.1 i.2.1.isLt (Nat.le_of_lt hk)).1
    simp only [loadA,naturalA,Bool.and_eq_true,Bool.or_eq_true,
      mask,row_mask x legal env _ _ hp i.1.isLt,toNat small,address,toNat (by omega :
        selectedNaturalRow x.M x.N x.config wrapped (s.pids 0) i.1.val*x.K+(k*x.config.blockK+i.2.1.val)<2^32)]
  · funext i
    have mask := inner_mask x.M x.N x.K x.config legal env k i.1.val (Nat.le_of_lt hk) i.1.isLt
    simp only [remaining] at mask
    have address := (offset_words x.M x.N x.K x.config legal env wrapped (s.pids 0) 0 i.2.1.val i.1.val k hp
      (Contract.positive_tiles x.config legal).1 i.2.1.isLt).2.1
    have bound := (offset_bounds x.M x.N x.K x.config legal env wrapped (s.pids 0) 0 i.2.1.val i.1.val k hp
      (Contract.positive_tiles x.config legal).1 i.2.1.isLt i.1.isLt (Nat.le_of_lt hk)).2.1
    simp only [loadB,naturalB,Bool.and_eq_true,Bool.or_eq_true,
      mask,column_mask x legal env _ _ hp i.2.1.isLt,toNat small,address,toNat (by omega :
        (k*x.config.blockK+i.1.val)*x.N+selectedNaturalColumn x.M x.N x.config wrapped (s.pids 0) i.2.1.val<2^32)]

def accumulation (x : Launch) (wrapped : Bool) (s : State Word) : Nat → CIndex x → Int
  | 0 => fun _ => 0
  | k+1 => fun i => BlockWord.wrap 32 (accumulation x wrapped s k i+
      (FP.IntegerDot.wordDot [] (naturalA x wrapped s k) (naturalB x wrapped s k) i).toInt)

def loop (x : Launch) (wrapped : Bool) (s : State Word) :
    Nat → BitVec 32 → (CIndex x → Int) → Option (CIndex x → Int)
  | 0,k,carry => if k.slt (cdiv (word x.K) (word x.config.blockK)) then none else some carry
  | fuel+1,k,carry => if k.slt (cdiv (word x.K) (word x.config.blockK)) then
      loop x wrapped s fuel (k+word 1) (fun i => BlockWord.wrap 32 (carry i+
        (FP.IntegerDot.wordDot [] (loadA x wrapped s k) (loadB x wrapped s k) i).toInt))
    else some carry

theorem loop_test (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (k : Nat) (hk : k≤steps x) :
    (word k).slt (cdiv (word x.K) (word x.config.blockK))=Bool.true ↔ k<steps x := by
  have bk := (Contract.positive_tiles x.config legal).2.2.1
  have eb := (Contract.envelope_fields x.M x.N x.K x.config env).2.2.1
  have ks := (ceil_small x.K x.config.blockK bk eb).2.2.1
  rw [cdiv_exact _ _ bk eb (tiles_small x.config legal).2.2]
  exact slt _ _ (lt_of_le_of_lt hk ks) ks

theorem loop_eq (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (wrapped : Bool)
    (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) (fuel k : Nat)
    (sum : k+fuel=steps x) :
    loop x wrapped s fuel (word k) (accumulation x wrapped s k)=some (accumulation x wrapped s (steps x)) := by
  induction fuel generalizing k with
  | zero =>
    have stop : k=steps x := by omega
    rw [loop,if_neg ((loop_test x legal env k (by omega)).not.mpr (by omega)),stop]
  | succ fuel ih =>
    have active : k<steps x := by omega
    rw [loop,if_pos ((loop_test x legal env k (by omega)).mpr active),
      (loads x legal env wrapped s hp k active).1,(loads x legal env wrapped s hp k active).2]
    change loop x wrapped s fuel (word k+word 1) (accumulation x wrapped s (k+1))=_
    rw [←BitVec.ofNat_add]
    exact ih (k+1) (by omega)

def active (x : Launch) (s : State Word) (i : CIndex x) : Prop :=
  naturalRow x.M x.N x.config (s.pids 0) i.1.val<x.M ∧
    naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val<x.N
instance (x : Launch) (s : State Word) (i : CIndex x) : Decidable (active x s i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def offset (x : Launch) (s : State Word) (i : CIndex x) :=
  naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N+
    naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val
abbrev Suffix (x : Launch) := (CIndex x → Int) → CIndex x → Int

def written (x : Launch) (wrapped : Bool) (s : State Word) (suffix : Suffix x) : State Word :=
  (TileShape.allIndices [x.config.blockM,x.config.blockN]).foldl
    (fun t i => if active x s i then t.write x.c (offset x s i)
      (.mk .int (suffix (accumulation x wrapped s (steps x)) i)) else t) s

def program (x : Launch) (wrapped : Bool) (s : State Word) (suffix : Suffix x) : Option (State Word) :=
  (loop x wrapped s (cdiv (word x.K) (word x.config.blockK)).toNat (word 0) (fun _ => 0)).map fun carry =>
    (TileShape.allIndices [x.config.blockM,x.config.blockN]).foldl
      (fun t i => if (row x.M x.N x.config (s.pids 0) i.1.val).slt (word x.M) &&
          (column x.M x.N x.config (s.pids 0) i.2.1.val).slt (word x.N) then
        t.write x.c (cOffset x.M x.N x.config (s.pids 0) i.1.val i.2.1.val).toNat
          (.mk .int (suffix carry i)) else t) s

theorem program_eq (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (wrapped : Bool)
    (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) (suffix : Suffix x) :
    program x wrapped s suffix=some (written x wrapped s suffix) := by
  have bk := (Contract.positive_tiles x.config legal).2.2.1
  have eb := (Contract.envelope_fields x.M x.N x.K x.config env).2.2.1
  have ks := (ceil_small x.K x.config.blockK bk eb).2.2.1
  have hl := loop_eq x legal env wrapped s hp (steps x) 0 (by omega)
  change loop x wrapped s (Contract.ceil x.K x.config.blockK) (word 0) (fun _ => 0)=_ at hl
  simp only [program,cdiv_exact _ _ bk eb (tiles_small x.config legal).2.2,
    toNat (by omega : Contract.ceil x.K x.config.blockK<2^32),hl,Option.map_some,written]
  congr 2
  funext t i
  have address := (offset_words x.M x.N x.K x.config legal env wrapped (s.pids 0) i.1.val i.2.1.val 0 0 hp
    i.1.isLt i.2.1.isLt).2.2
  have bound := (offset_bounds x.M x.N x.K x.config legal env wrapped (s.pids 0) i.1.val i.2.1.val 0 0 hp
    i.1.isLt i.2.1.isLt bk (Nat.zero_le _)).2.2
  simp only [Bool.and_eq_true,row_mask x legal env _ _ hp i.1.isLt,
    column_mask x legal env _ _ hp i.2.1.isLt,active,address,offset,
    toNat (by omega : naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N+
      naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val<2^32)]

def intProgram (x : Launch) (s : State Word) := program x Bool.false s (fun carry i => carry i)

def scaledSuffix (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch)
    (scale : RegionName) (s : State Word) : Suffix x := fun carry i =>
  FP.IntegerConversion.totalSigned (u 32) 32
    (FP.mul cfg (FP.round .fp32 cfg (carry i : Rat))
      ((s.mem scale (row x.M x.N x.config (s.pids 0) i.1.val).toNat).read .real))
def scaledProgram (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch)
    (scale : RegionName) (s : State Word) := program x Bool.true s (scaledSuffix cfg u x scale s)

theorem source_loop_test (x : Launch) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (k : Nat) (hk : k≤steps x) :
    ((word k).slt (cdiv (word x.K) (word x.config.blockK))=Bool.true ↔ k*x.config.blockK<x.K) ∧
    ((word 0).slt (remaining x.K x.config k)=Bool.true ↔ k<steps x) := by
  have natural := BlockWord.before_stop (blockParameters x) (Contract.positive_tiles x.config legal).2.2.1 k
  exact ⟨(loop_test x legal env k hk).trans natural,
    (descending_test x.M x.N x.K x.config legal env k hk).trans natural.symm⟩

#axiomsClean loads
#axiomsClean loop_eq
#axiomsClean program_eq
#axiomsClean source_loop_test
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteExecution
