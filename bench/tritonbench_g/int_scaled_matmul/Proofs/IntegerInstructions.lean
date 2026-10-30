import bench.tritonbench_g.int_scaled_matmul.Proofs.SaturatingTile
import bench.tritonbench_g.int_scaled_matmul.Proofs.FiniteGrid

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.IntegerInstructions
open VeriTile.Triton FP.Structural
open FiniteExecution FiniteIndex
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Fixed 32-product MMA groups; a zero-length terminal group is an identity
and represents no machine instruction. A short group is zero-padded. -/
def partition (n : Nat) : List Nat := List.replicate (n/32) 32 ++ [n%32]
theorem partition_cover (n : Nat) : (partition n).sum=n := by
  simp only [partition,List.sum_append,List.sum_replicate,List.sum_cons,List.sum_nil,Nat.add_zero]
  change n / 32 * 32 + n % 32 = n
  have h := Nat.div_add_mod n 32
  omega

def dot (x : Launch) (wrapped : Bool) (s : State Word) (k : BitVec 32) (i : CIndex x) : Int :=
  SaturatingTile.mmaBlocks (SaturatingTile.tileProduct x.config (loadA x wrapped s k) (loadB x wrapped s k) i)
    (partition x.config.blockK) 0 0

theorem dot_eq (x : Launch) (legal : Contract.ConfigContract x.config)
    (wrapped : Bool) (s : State Word) (k : BitVec 32) (i : CIndex x) :
    dot x wrapped s k i=(FP.IntegerDot.wordDot [] (loadA x wrapped s k) (loadB x wrapped s k) i).toInt := by
  symm
  exact SaturatingTile.source_dot_eq x.config legal _ _
    (fun _ => SaturatingTile.wrap_signed8 _) (fun _ => SaturatingTile.wrap_signed8 _) i _ (partition_cover _)

def loop (x : Launch) (wrapped : Bool) (s : State Word) :
    Nat → BitVec 32 → (CIndex x → Int) → Option (CIndex x → Int)
  | 0,k,carry => if k.slt (cdiv (word x.K) (word x.config.blockK)) then none else some carry
  | fuel+1,k,carry => if k.slt (cdiv (word x.K) (word x.config.blockK)) then
      loop x wrapped s fuel (k+word 1) (fun i =>
        (BitVec.ofInt 32 (carry i)+BitVec.ofInt 32 (dot x wrapped s k i)).toInt)
    else some carry

theorem loop_eq (x : Launch) (legal : Contract.ConfigContract x.config)
    (wrapped : Bool) (s : State Word) (fuel : Nat) (k : BitVec 32) (carry : CIndex x → Int) :
    loop x wrapped s fuel k carry=FiniteExecution.loop x wrapped s fuel k carry := by
  induction fuel generalizing k carry with
  | zero => rfl
  | succ fuel ih =>
    simp only [loop,FiniteExecution.loop,ih,dot_eq x legal,SaturatingTile.add_u32]

def program (x : Launch) (wrapped : Bool) (s : State Word) (suffix : Suffix x) : Option (State Word) :=
  (loop x wrapped s (cdiv (word x.K) (word x.config.blockK)).toNat (word 0) (fun _ => 0)).map fun carry =>
    (TileShape.allIndices [x.config.blockM,x.config.blockN]).foldl
      (fun t i => if (row x.M x.N x.config (s.pids 0) i.1.val).slt (word x.M) &&
          (column x.M x.N x.config (s.pids 0) i.2.1.val).slt (word x.N) then
        t.write x.c (cOffset x.M x.N x.config (s.pids 0) i.1.val i.2.1.val).toNat
          (.mk .int (suffix carry i)) else t) s

theorem program_eq (x : Launch) (legal : Contract.ConfigContract x.config)
    (wrapped : Bool) (s : State Word) (suffix : Suffix x) :
    program x wrapped s suffix=FiniteExecution.program x wrapped s suffix := by
  simp only [program,FiniteExecution.program,loop_eq x legal]

def intOne (x : Launch) (pid : Nat) (s : State Word) :=
  program x Bool.false (GridExecution.atProgram s pid) (fun carry i => carry i)
def scaledOne (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName)
    (pid : Nat) (s : State Word) :=
  let current := GridExecution.atProgram s pid
  program x Bool.true current (scaledSuffix cfg u x scale current)
def intGrid (x : Launch) (s : State Word) :=
  GridExecution.sequence (intOne x) (List.range (count x.M x.N x.config)) s
def scaledGrid (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName) (s : State Word) :=
  GridExecution.sequence (scaledOne cfg u x scale) (List.range (count x.M x.N x.config)) s

theorem intGrid_eq (x : Launch) (legal : Contract.ConfigContract x.config) (s : State Word) :
    intGrid x s=FiniteGrid.intGrid x s := by
  have one : intOne x=FiniteGrid.intOne x := by
    funext pid s
    exact program_eq x legal Bool.false _ _
  simp only [intGrid,FiniteGrid.intGrid,one]

theorem scaledGrid_eq (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName)
    (legal : Contract.ConfigContract x.config) (s : State Word) :
    scaledGrid cfg u x scale s=FiniteGrid.scaledGrid cfg u x scale s := by
  have one : scaledOne cfg u x scale=FiniteGrid.scaledOne cfg u x scale := by
    funext pid s
    exact program_eq x legal Bool.true _ _
  simp only [scaledGrid,FiniteGrid.scaledGrid,one]

#axiomsClean dot_eq
#axiomsClean loop_eq
#axiomsClean program_eq
#axiomsClean intGrid_eq
#axiomsClean scaledGrid_eq
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.IntegerInstructions
