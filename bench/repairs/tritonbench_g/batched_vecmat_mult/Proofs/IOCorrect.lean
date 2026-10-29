import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOSafety
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOResults

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.IOCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open VecmatIO
noncomputable section
set_option maxHeartbeats 2000000

theorem graph (rest : Algebra Expr) (profile : KernelProfile.Selector) (p : Parameters)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0<p.BM) (hBN : 0<p.BN)
    (hM : p.BM∣p.M) (hN : p.BN∣p.N) :
    (graphIO rest profile p).Implements (graphTarget profile p) := by
  intro vals _ bounds hb initial hin
  have windows : IOSafety.Windows p bounds :=
    ⟨fun m k => hb.2.1 (0 : Fin 2) (m,k,()) trivial,
      fun m n k => hb.2.1 (1 : Fin 2) (m,n,k,()) trivial,
      fun m n => hb.2.2.1 (0 : Fin 1) (m,n,()) trivial⟩
  obtain ⟨final,ran,hm⟩ := IOSchedule.grid rest profile p initial
  refine ⟨IOSafety.grid rest profile p bounds windows initial,final,ran,?_,?_⟩
  · intro o j _
    change Fin 1 at o
    have eq : o=(0 : Fin 1) := Subsingleton.elim _ _
    subst o
    change mapCell id (final.mem p.output (j.1.val*p.N+j.2.1.val))=
      Cell.mk .real (graphTarget profile p vals (0 : Fin 1) j)
    rw [IOResults.output profile p initial final separate hBM hBN hM hN hm j.1 j.2.1]
    apply congrArg (Cell.mk .real)
    exact IOResults.tensor_formula (KernelProfile.toPlans profile) p (fun _ _ => id) vals initial.mem
      (fun m k => hin (0 : Fin 2) (m,k,()) trivial)
      (fun m n k => hin (1 : Fin 2) (m,n,k,()) trivial) j.1 j.2.1 _ (by rfl)
  · intro r address outside
    exact IOResults.frame profile p initial final hm r address
      (fun m n => outside.1 (0 : Fin 1) (m,n,()) trivial)

theorem real (decode : ValueProofs.Word → ℝ) (rest : Algebra Expr)
    (profile : KernelProfile.Selector) (p : Parameters)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0<p.BM) (hBN : 0<p.BN)
    (hM : p.BM∣p.M) (hN : p.BN∣p.N) (hK : p.BK∣p.K) :
    (graphIO rest profile p).Observes
      (fun _ _ e => SourceGraph.realValue decode (SourceGraph.contract e))
      (realTarget decode p) := by
  intro vals _ bounds hb initial hin
  have windows : IOSafety.Windows p bounds :=
    ⟨fun m k => hb.2.1 (0 : Fin 2) (m,k,()) trivial,
      fun m n k => hb.2.1 (1 : Fin 2) (m,n,k,()) trivial,
      fun m n => hb.2.2.1 (0 : Fin 1) (m,n,()) trivial⟩
  obtain ⟨final,ran,hm⟩ := IOSchedule.grid rest profile p initial
  refine ⟨IOSafety.grid rest profile p bounds windows initial,final,ran,?_,?_⟩
  · intro o j _
    change Fin 1 at o
    have eq : o=(0 : Fin 1) := Subsingleton.elim _ _
    subst o
    change mapCell (fun e => SourceGraph.realValue decode (SourceGraph.contract e))
        (mapCell id (final.mem p.output (j.1.val*p.N+j.2.1.val))) =
      Cell.mk .real (realTarget decode p vals (0 : Fin 1) j)
    rw [IOResults.output profile p initial final separate hBM hBN hM hN hm j.1 j.2.1]
    apply congrArg (Cell.mk .real)
    exact IOResults.real_formula decode profile p vals initial.mem
      (fun m k => hin (0 : Fin 2) (m,k,()) trivial)
      (fun m n k => hin (1 : Fin 2) (m,n,k,()) trivial) hK j.1 j.2.1
  · intro r address outside
    exact IOResults.frame profile p initial final hm r address
      (fun m n => outside.1 (0 : Fin 1) (m,n,()) trivial)

/-- Addressed word inputs are an additional public specialization. The wider
arbitrary-expression interface above also admits prior mismatched-tag defaults. -/
theorem native (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (separate : SourceGraph.InputsSeparate p)
    (hBM : 0<p.BM) (hBN : 0<p.BN) (hM : p.BM∣p.M) (hN : p.BN∣p.N) :
    (nativeIO cfg rest profile p).Implements (nativeTarget cfg profile p) := by
  apply AddressedProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply IOSafety.grid rest profile p bounds
    exact ⟨fun m k => hb.2.1 (0 : Fin 2) (m,k,()) trivial,
      fun m n k => hb.2.1 (1 : Fin 2) (m,n,k,()) trivial,
      fun m n => hb.2.2.1 (0 : Fin 1) (m,n,()) trivial⟩
  · intro vals initial hin
    obtain ⟨final,ran,hm⟩ := IOSchedule.grid rest profile p initial
    refine ⟨final,ran,?_,?_⟩
    · intro o j _
      change Fin 1 at o
      have eq : o=(0 : Fin 1) := Subsingleton.elim _ _
      subst o
      change mapCell (fun e => SourceGraph.evaluate cfg (SourceGraph.contract e))
          (final.mem p.output (j.1.val*p.N+j.2.1.val))=
        Cell.mk .real (nativeTarget cfg profile p vals (0 : Fin 1) j)
      rw [IOResults.output profile p initial final separate hBM hBN hM hN hm j.1 j.2.1]
      apply congrArg (fun e => Cell.mk .real (SourceGraph.evaluate cfg (SourceGraph.contract e)))
      exact IOResults.tensor_formula (KernelProfile.toPlans profile) p Expr.input vals initial.mem
        (fun m k => hin (0 : Fin 2) (m,k,()) trivial)
        (fun m n k => hin (1 : Fin 2) (m,n,k,()) trivial) j.1 j.2.1 _ (by rfl)
    · intro r address outside
      exact IOResults.frame profile p initial final hm r address
        (fun m n => outside.1 (0 : Fin 1) (m,n,()) trivial)

#axiomsClean graph
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.IOCorrect
