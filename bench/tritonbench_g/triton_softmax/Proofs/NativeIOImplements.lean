import bench.tritonbench_g.triton_softmax.IO
import bench.tritonbench_g.triton_softmax.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.TritonSoftmax.NativeIOImplements
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section
open IO

theorem correct (inst : NativeGraph.Instructions) (candidate : Bool) (x y : View)
    (B p : Nat) (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows) :
    (nativeIO inst candidate x y B p).Implements
      (IO.nativeValues inst candidate x y B p (NativeIO.reference inst x.cols B)) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    have hin : ∀ i : Fin B,i.val<x.cols → x.index p i.val<bounds x.region :=
      fun i hi => hb.2.1 (0 : Fin 1) (i,PUnit.unit) hi
    have hout : ∀ i : Fin B,i.val<x.cols → y.index p i.val<bounds y.region :=
      fun i hi => hb.2.2.1 (0 : Fin 1) (i,PUnit.unit) hi
    cases candidate
    · exact (scheduleSafe_singleton ..).mpr (StructuralAccess.source_safe _ _ _ _ _ _ _ _ _ _ _ positive bounds
        (Floating.launchState x.descriptor p s) hin hout)
    · exact (scheduleSafe_singleton ..).mpr (StructuralAccess.candidate_safe _ _ _ _ _ _ _ _ _ _ _ positive bounds
        (Floating.launchState x.descriptor p s) hin hout)
  · intro xs s input
    obtain ⟨t,run,values,frame⟩ := NativeIO.result inst x y B p shape injective positive cover row s
      (fun j => xs (0 : Fin 1) (j,PUnit.unit)) (fun j hj => input (0 : Fin 1) (j,PUnit.unit) hj)
    refine ⟨t,?_,?_,?_⟩
    · cases candidate
      · simpa [nativeIO,runSchedule,Invocation.run,Invocation.initial,bindRegisters,Floating.evaluate,
          Floating.launchState,SoftmaxTriton1.Floating.launchState,View.descriptor] using run
      · simpa [nativeIO,runSchedule,Invocation.run,Invocation.initial,bindRegisters,KernelNative.candidateEvaluate,
          Floating.launchState,SoftmaxTriton1.Floating.launchState,View.descriptor] using (KernelNative.execution (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p s).symm.trans run
    · intro o i hi
      fin_cases o
      exact values i.1 hi
    · intro r off miss
      apply frame r off
      intro i hi eq
      have outside := miss.1 (0 : Fin 1) (i,PUnit.unit) hi
      exact outside.elim (fun h => h eq.1.symm) (fun h => h eq.2.symm)


#axiomsClean correct
end
end VeriTile.Bench.Repairs.TritonSoftmax.NativeIOImplements
