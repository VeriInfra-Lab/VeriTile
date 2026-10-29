import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.NativeGraph
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPWrapper

namespace VeriTile.Bench.Repairs.ReversedCumsum.NativeWrapper
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section

def logical (words : Nat → Nat → Nat → BitVec 32) : Nat → Nat → Nat → Expr :=
  fun row t s => .leaf (words row t s)

def models (rest : Algebra Expr) : Config → Algebra Expr := NativeGraph.algebra rest

def reference (rest : Algebra Expr) (h : Call) (c : Config) (positive : 0<c.BT)
    (words : Nat → Nat → Nat → BitVec 32) : Nat → Nat → Nat → Cell Expr :=
  FPWrapper.reference (models rest c) (NativeGraph.backend c) h c positive (logical words)

theorem reference_materialized (rest : Algebra Expr) (h : Call) (outType : h.outputDType=.fp32)
    (c : Config) (positive : 0<c.BT) (words : Nat → Nat → Nat → BitVec 32) (row t s : Nat) :
    NativeGraph.Materialized ((reference rest h c positive words row t s).read .fp32) := by
  have materialized := NativeGraph.expected_materialized rest h c positive (s/BS) (logical words row)
    (fun _ _ _ _ => True.intro) t ⟨s%BS,Nat.mod_lt _ (by decide)⟩
  unfold reference FPWrapper.reference FPMemory.stored
  rw [outType]
  exact materialized

#axiomsClean reference_materialized
end
end VeriTile.Bench.Repairs.ReversedCumsum.NativeWrapper
