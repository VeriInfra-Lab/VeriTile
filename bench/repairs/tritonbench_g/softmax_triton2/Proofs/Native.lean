import bench.repairs.tritonbench_g.softmax_triton2.Proofs.Binding
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.NativeProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous NativeGraph
noncomputable section

/-- Execute this task's shared typed source with the ordered FP32 graph
algebra, then project each stored word from its owning thread. Production
layout V1 and the six scalar instruction ports are explicit model boundaries. -/
def evaluate (inst : Instructions) (x y : Tensor) (B p : Nat)
    (s : State Word) : Option (State Word) :=
  (exec (Floating.backend (algebra inst 1))
    (SoftmaxTriton2.softmax_kernel y.region x.region x.offset y.offset
      x.rowStride x.colStride y.rowStride y.colStride x.cols B)
    (Floating.launchState x p (liftState s))).map (projectState 1 B y p s)

/-- This identity is between actual evaluators, not between two failures. -/
theorem evaluator_identity (inst : Instructions) (x y : Tensor) (B p : Nat)
    (s : State Word) : evaluate inst x y B p s = NativeGraph.evaluate inst 1 x y B p s := rfl

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

theorem checked (inst : Instructions) (x : Object) (h : InputContract x.metadata)
    (y : Tensor) (shape : SameShape (descriptor x) y) (injective : y.Injective)
    (p : Nat) (hp : p < rows x.metadata)
    (read : Nat → Word) (s : State Word) (binding : InputMemory Floating.access read x s) :
    ∃ t, evaluate inst (descriptor x) y (blockSize (cols x.metadata)) p s = some t ∧
      (∀ c, c < cols x.metadata →
        (t.mem y.region (y.index p c)).read .real =
          NativeGraph.row inst 1 (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue negateWord read x p) c) ∧
      FrameRows Floating.access y [p] s t := by
  have input : Inputs Floating.access (descriptor x) (logicalValue negateWord read x) s := by
    intro r c hr hc
    simpa [descriptor,Tensor.index] using input_read Floating.access negateWord read x h s binding r c hr hc
  obtain ⟨t,he⟩ := NativeGraph.source_terminates inst 1 (descriptor x) y
    (blockSize (cols x.metadata)) p (blockSize_bounds _).1 s
  have result := (NativeGraph.source inst 1).row (descriptor x) y
    (blockSize (cols x.metadata)) p s t (logicalValue negateWord read x)
    shape injective (blockSize_bounds _).1 (blockSize_bounds _).2 hp input he
  exact ⟨t,he,result⟩

#axiomsClean evaluator_identity
#axiomsClean checked
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.NativeProof
