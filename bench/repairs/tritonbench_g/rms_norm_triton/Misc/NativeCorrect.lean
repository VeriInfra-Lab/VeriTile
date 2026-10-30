import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.RuntimeObserved
import bench.repairs.tritonbench_g.rms_norm_triton.IO
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RmsNormTriton.FPEquiv
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural Whole Observed
noncomputable section

/-- The actual source executes its ordered FP32 graph. Input words are encoded
as input nodes; each output lane observes the native graph with that lane's
reduction tree and the actual runtime epsilon word. -/
abbrev nativeIO := FloatingIO.nativeIO

/-- Independent native recurrence retains local instructions, reduction trees,
FMA contraction and runtime epsilon. No FP-to-Real identity is asserted. -/
def formula (M : NativeModel (BitVec 32)) (profile : ReductionFamily) (N B : Nat)
    (xs ws : Nat → BitVec 32) (i : Nat) : BitVec 32 := RuntimeObserved.reference M profile N B xs ws i

def expected (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat)
    (xs : (nativeIO c eps M profile rows).Inputs) : (nativeIO c eps M profile rows).Outputs :=
  fun o => Fin.cases (fun j => formula M profile c.N c.B
    (GridPublic.inputRow graphAlgebra Term.input (interpret M profile) c eps rows xs j.1)
    (GridPublic.weights graphAlgebra Term.input (interpret M profile) c eps rows xs) j.2.1.val)
    (fun z => Fin.elim0 z) o

/-- Safe successful full-grid source execution, independent native words, and
complete exterior frame. Primitive instructions and reduction topology are
model boundaries; this theorem assumes no numerical transformation laws. -/
theorem kernel_native_equivalence (c : Layout) (eps : ℝ)
    (M : NativeModel (BitVec 32)) (epsilonWord : BitVec 32)
    (_epsilonArgument : M.epsilon=epsilonWord) (profile : ReductionFamily)
    (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) :
    (nativeIO c eps M profile rows).Implements (expected c eps M profile rows) := by
  exact GridPublic.implements_argument graphAlgebra Term.input (interpret M profile) c eps (.parameter "eps") rows schedule
    xSeparate wSeparate (formula M profile c.N c.B) (RuntimeObserved.source_graph_lowering M profile c)

#specNonCircular formula avoiding [RmsNormTriton.kernel,lower,interpret,rowValue,sourceRun,runRows]
#axiomsClean kernel_native_equivalence
#stmtSurfaceSubset kernel_native_equivalence ⊆
  [Layout,Layout.X,Layout.W,Layout.Y,Layout.M,RegionName,NativeModel,NativeModel.epsilon,
   ReductionFamily,Term,instInhabitedTerm,nativeIO,expected,ProgramIO.Implements]
end
end VeriTile.Bench.Repairs.RmsNormTriton.FPEquiv
