import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelRealReference
open VeriTile.Triton FP.Structural KernelProgram
open scoped BigOperators
noncomputable section
open Classical
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
namespace Formula
export TritonBenchG.ChunkGateRecurrence.RealMath (Inputs initial forward reverse forward_zero forward_step reverse_end reverse_step)
end Formula

def reduce {shape : TileShape} (axisIndex : Fin shape.length) (keep : Bool) (values : Values ℝ .real shape) :
    Values ℝ .real (TileShape.reduceShape shape axisIndex keep) :=
  if hs : shape=[1024] then
    let v : Values ℝ .real [1024] := hs ▸ values
    fun _ => ∑ i : Fin 1024,v (i,PUnit.unit)
  else fun _ => 0
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .mul => a*b
    | .add => a+b
    | .sub => a-b
    | _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} => reduce
def inputs (a : KernelReference.Inputs ℝ) : Formula.Inputs 64 16 where
  source := fun t e j => a.source t (e,j,PUnit.unit)
  gate := a.gate
  seed := fun e j => a.seed (e,j,PUnit.unit)
  incoming := fun t e j => a.incoming t (e,j,PUnit.unit)
def index (i : Fin 1024) : Lane :=
  TileShape.reshapeIndex (input:=[64,16]) (output:=[64*16]) (by decide) (i,PUnit.unit)
def contraction (carry saved : Lane → ℝ) : ℝ := ∑ i : Fin 1024,carry (index i)*saved (index i)

theorem forward_formula (a : KernelReference.Inputs ℝ) (seed : Bool) (n : Nat)
    (e : Fin 64) (j : Fin 16) :
    KernelReference.forward algebra a seed n (e,j,PUnit.unit)=Formula.forward (inputs a) seed n e j := by
  induction n with
  | zero => rw [Formula.forward_zero]; rfl
  | succ n ih =>
    rw [Formula.forward_step,KernelReference.forward]
    change KernelReference.forward algebra a seed n (e,j,PUnit.unit)*a.gate n+a.source n (e,j,PUnit.unit)=_
    rw [ih]
    rfl
theorem reverse_formula (a : KernelReference.Inputs ℝ) (N n : Nat) (hn : n≤N)
    (e : Fin 64) (j : Fin 16) :
    KernelReference.reverse algebra a N n (e,j,PUnit.unit)=Formula.reverse (inputs a) N (N-n) e j := by
  induction n with
  | zero => simp [KernelReference.reverse,KernelReference.zero,algebra]
  | succ n ih =>
    have ti : N-(n+1)=N-1-n := by omega
    have si : N-n=(N-1-n)+1 := by omega
    rw [ti,Formula.reverse_step _ _ _ (by omega),KernelReference.reverse]
    change KernelReference.reverse algebra a N n (e,j,PUnit.unit)*a.gate (N-1-n)+a.incoming (N-1-n) (e,j,PUnit.unit)=_
    rw [ih (by omega),si]
    rfl
theorem contraction_formula (carry saved : Lane → ℝ) :
    KernelReference.contraction algebra carry saved=contraction carry saved := by
  simp only [KernelReference.contraction,KernelReference.flattened,KernelReference.product,algebra,reduce,dif_pos rfl]
  rfl

#axiomsClean forward_formula
#axiomsClean reverse_formula
#axiomsClean contraction_formula
#specNonCircular Formula.forward avoiding [KernelReference.forward,FP.Structural.exec,FP.Structural.run]
#specNonCircular Formula.reverse avoiding [KernelReference.reverse,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelRealReference
