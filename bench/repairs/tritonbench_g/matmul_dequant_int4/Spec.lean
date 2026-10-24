import bench.repairs.tritonbench_g.matmul_dequant_int4.Kernels
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
noncomputable section

/-- Base-16 digit interpretation independent of signed source shift operations. -/
def digit (word : Word) (position : Nat) : Int :=
  ((word.toNat / 2^(4*(position%8)))%16 : Nat)
def logicalQ (s : Shape) (initial : State α) (k n : Nat) :=
  digit (initial.packed .weight ((k/8)*s.n+n)) k
def logicalZ (s : Shape) (initial : State α) (k n : Nat) :=
  digit (initial.packed .zero ((k/s.group)*(s.n/8)+n/8)) n
def logicalScale (s : Shape) (initial : State α) (k n : Nat) :=
  initial.values .scale ((k/s.group)*s.n+n)
def logicalA (s : Shape) (initial : State α) (r k : Nat) :=
  initial.values .a (r*s.k+k)
def dequantizedReal (s : Shape) (initial : State ℝ) (k n : Nat) : ℝ :=
  ((logicalQ s initial k n-logicalZ s initial k n : Int):ℝ)*logicalScale s initial k n
def sumReal (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => sumReal f n+f n
def mathematical (s : Shape) (initial : State ℝ) (r n : Nat) : ℝ :=
  sumReal (fun k => logicalA s initial r k*dequantizedReal s initial k n) s.k

def realDequant : DequantOps ℝ := ⟨(fun z => z),(·*·)⟩
def realFused : FusedOps ℝ ℝ where
  zeroInput := 0
  zeroAcc := 0
  fromInt := fun z => z
  mul := (·*·)
  sub := (·-·)
  neg := (-·)
  fma := fun x y z => x*y+z
  mma16 := fun a b seed => seed+sumReal (fun i => a ⟨i%16,Nat.mod_lt _ (by decide)⟩*b ⟨i%16,Nat.mod_lt _ (by decide)⟩) 16
  narrow := id

end
end VeriTile.Bench.Repairs.MatmulDequantInt4
