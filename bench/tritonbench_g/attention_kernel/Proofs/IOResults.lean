import bench.tritonbench_g.attention_kernel.Proofs.IOGridSafety
import bench.tritonbench_g.attention_kernel.Proofs.IORealFormula

namespace VeriTile.Bench.Repairs.AttentionKernel.IOResults
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem real_implements (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (hbm : 0<p.bm) (separate : Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    (IO.realIO p work).Implements (Target.mathematical p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    exact IOGridSafety.safe Real.algebra Real.dotModel p bases domain hbm work
      (fun a ha=>order.mem_iff.mpr ha) some (fun _ x=>x.unbotD 0) vals bounds bounded initial
  · intro vals before binding
    obtain ⟨after,ran,values,frame⟩:=RawKernel.complete Real.algebra Real.dotModel p hbm
      domain.launch separate before work order
    refine ⟨after,?_,?_,?_⟩
    · exact (IOGridSafety.schedule Real.algebra Real.dotModel p work before).trans ran
    · intro o j _
      change Fin 1 at o
      have hz : o=0 := Fin.eq_zero o
      subst o
      rcases j with ⟨b,h,n,d,u⟩
      cases u
      change mapCell (fun x : Real.R=>x.unbotD 0)
          (after.mem p.Out (logicalAddress p b.val h.val n.val d.val))=
        .mk p.storage.tile (ofFloat p.storage.float
          (RealMath.expected p.length p.dim p.bn p.scale
            (Target.views p vals b h).q (Target.views p vals b h).k
            (Target.views p vals b h).v (Target.views p vals b h).bias n.val d.val))
      rw [values b.val b.isLt h.val h.isLt n.val n.isLt d.val d.isLt]
      have loads:=IOMemoryBinding.reads Real.algebra Real.dotModel p work
        some (fun _ x=>x.unbotD 0) vals before binding b h
      have guard:=domain.launch ⟨lt_of_le_of_lt (Nat.zero_le _) b.isLt,
        lt_of_le_of_lt (Nat.zero_le _) h.isLt,lt_of_le_of_lt (Nat.zero_le _) n.isLt⟩
      rw [IORealFormula.answer_eq p guard hbm vals before b h n d loads,IOMemoryBinding.map_output]
      rfl
    · intro r off outside
      exact frame r off (IOMemoryBinding.exterior Real.algebra Real.dotModel p work
        some (fun _ x=>x.unbotD 0) vals r off outside)

theorem native_implements (I : KernelOnly.Instructions) (layout : FamilyNative.Layout)
    (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (hbm : 0<p.bm) (separate : Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    (IO.nativeIO I layout p work).Implements (Target.native I layout p hbm) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    exact IOGridSafety.safe (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
      p bases domain hbm work (fun a ha=>order.mem_iff.mpr ha)
      IO.encodeWord (IO.observeWord I layout p) vals bounds bounded initial
  · intro vals before binding
    obtain ⟨after,ran,values,frame⟩:=RawKernel.complete (FamilyNative.algebra layout)
      (FamilyNative.dotModel layout) p hbm domain.launch separate before work order
    refine ⟨after,?_,?_,?_⟩
    · exact (IOGridSafety.schedule (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
        p work before).trans ran
    · intro o j _
      change Fin 1 at o
      have hz : o=0 := Fin.eq_zero o
      subst o
      rcases j with ⟨b,h,n,d,u⟩
      cases u
      change mapCell (IO.observeWord I layout p (b,h,n,d,()))
          (after.mem p.Out (logicalAddress p b.val h.val n.val d.val))=
        .mk p.storage.tile (ofFloat p.storage.float
          (IO.observeWord I layout p (b,h,n,d,())
            (Target.Native.algorithm (FamilyNative.algebra layout) (FamilyNative.dotModel layout) p
              (IOMemoryBinding.encodedViews IO.encodeWord p vals b h)
              (n.val/p.bm) (lane p hbm n.val d.val d.isLt))))
      rw [values b.val b.isLt h.val h.isLt n.val n.isLt d.val d.isLt]
      have loads:=IOMemoryBinding.reads (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
        p work IO.encodeWord (IO.observeWord I layout p) vals before binding b h
      have guard:=domain.launch ⟨lt_of_le_of_lt (Nat.zero_le _) b.isLt,
        lt_of_le_of_lt (Nat.zero_le _) h.isLt,lt_of_le_of_lt (Nat.zero_le _) n.isLt⟩
      rw [IOFormulaBinding.source_at (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
        p before (n.val/p.bm) b.val h.val h.isLt _ loads guard,IOMemoryBinding.map_output]
    · intro r off outside
      exact frame r off (IOMemoryBinding.exterior (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
        p work IO.encodeWord (IO.observeWord I layout p) vals r off outside)

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.AttentionKernel.IOResults
