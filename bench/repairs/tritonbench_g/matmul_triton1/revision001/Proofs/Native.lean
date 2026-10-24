import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Host
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Native
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def sourceConfig : FP.Config := FP.Config.ieeeValue
/-- One scalar FP32 fused multiply-add. The source explicitly disables TF32;
the selected SM90 lowering contains fma.rn.f32 and no MMA/TF32 operation. -/
structure Instructions where
  zero : Word
  fma : Word → Word → Word → Word

def referenceInstructions : Instructions :=
  ⟨FP.Value.zero .fp32,FP.fma sourceConfig⟩

/-- Only zero and one scalar FMA are local numerical obligations. The fixed
32-term source dot, all K blocks, grid, allocation, output and frame are derived. -/
structure Correspondence (actual : Instructions) : Prop where
  zero : actual.zero=FP.Value.zero .fp32
  fma : ∀ a b seed, actual.fma a b seed=FP.fma sourceConfig a b seed

def dot (I : Instructions) {n : Nat} (a b : Fin n → Word) (z : Word) : Word :=
  (List.finRange n).foldl (fun acc k => I.fma (a k) (b k) acc) z

def primitives (I : Instructions) : Core.Primitives Word where
  zero := I.zero
  dot := dot I

theorem correspondence_consistent : Correspondence referenceInstructions :=
  ⟨rfl,fun _ _ _ => rfl⟩

theorem instruction_eq (actual : Instructions) (numeric : Correspondence actual) :
    actual=referenceInstructions := by
  cases actual with
  | mk zero fma =>
    have hz := numeric.zero
    have hf : fma=FP.fma sourceConfig := by funext a b seed; exact numeric.fma a b seed
    simp only [referenceInstructions]
    congr

theorem dot_eq (actual : Instructions) (numeric : Correspondence actual)
    {n : Nat} (a b : Fin n → Word) (z : Word) :
    dot actual a b z=dot referenceInstructions a b z := by rw [instruction_eq actual numeric]

theorem all_inner_lanes (n : Nat) (k : Fin n) : k∈List.finRange n := by simp

theorem dot_step (I : Instructions) {n : Nat} (a b : Fin (n+1) → Word) (z : Word) :
    dot I a b z=I.fma (a (Fin.last n)) (b (Fin.last n))
      (dot I (fun i => a i.castSucc) (fun i => b i.castSucc) z) := by
  simp [dot,List.finRange_succ_last,List.foldl_append,List.foldl_map]

theorem host_eq (actual : Instructions) (numeric : Correspondence actual)
    (rest : Algebra Word) (allocate : Host.Allocator Word)
    (x : Host.Arguments) (s : State Word) :
    Host.call rest (primitives actual) allocate x s=
      Host.call rest (primitives referenceInstructions) allocate x s := by
  rw [instruction_eq actual numeric]

#axiomsClean correspondence_consistent
#axiomsClean instruction_eq
#axiomsClean dot_eq
#axiomsClean all_inner_lanes
#axiomsClean dot_step
#axiomsClean host_eq
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Native
