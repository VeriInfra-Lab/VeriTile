import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.tritonbench_g.matrix_vector_multip.Proofs.GridPair

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
open scoped VeriTile.Spec
noncomputable section

def gridOriginal (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (order : List Nat) :=
  ProgramIO.Implementation.original (GridPair.io native N M BN BM g order) (fun _ => True)

def gridExplicitFused (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (order : List Nat) :=
  ProgramIO.Implementation.ofVariant (GridPair.variant native N M BN BM g order) (fun _ => True)

/-- Full-N output interface and an entire actual grid order. The stronger
helper permits any order; this headline exposes exactly a complete source
grid, including its zero-program empty case and the final masked row tile. -/
specification grid_explicit_fma_equiv (native : Scalars) (N M BN BM : Nat)
    (g : Graph BN BM) (scope : SourceBounds N M BN BM) (rows : 0<BN) (width : 0<BM) (order : List Nat)
    (_schedule : order.Perm (List.range ((N+BN-1)/BN))) :
    gridOriginal native N M BN BM g order ≡[[]] gridExplicitFused native N M BN BM g order :=
  ProgramIO.Implementation.of_equivalent (GridPair.variant native N M BN BM g order)
    (GridPair.equivalent native N M BN BM g width scope order (by
      intro pid member
      exact scope.grid_pid pid (List.mem_range.mp (_schedule.mem_iff.mp member))))

#print_fp_assumptions grid_explicit_fma_equiv
#axiomsClean grid_explicit_fma_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
