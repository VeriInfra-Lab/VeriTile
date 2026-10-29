import bench.repairs.tritonbench_g.attention_kernel.Proofs.RawKernel
import bench.repairs.tritonbench_g.attention_kernel.Proofs.RealReturned
import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Binding

/-! Retained raw-memory and empty-grid consequences. Public IO results and
Tensor applications reuse the same `RawKernel.complete` execution foundation. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.KernelResults.Real
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine
noncomputable section
abbrev target := RealReturned.target

/-- Complete source grid and independent real attention on raw input memory. -/
theorem kernel_correct (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (hbm : 0<p.bm) (separate : Separated p) (before : State Real.R)
    (read : RegionName → Nat → ℝ) (loads : Real.Reads p before read)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.dim,
        after.mem p.Out (logicalAddress p b h n d)=
          .mk p.storage.tile (ofFloat p.storage.float (some (target p read b h n d)))) ∧
      (∀ r off,Physical.Exterior p r off → after.mem r off=before.mem r off) ∧ Closed p bases := by
  obtain ⟨after,run,values,frame⟩ := RawKernel.complete Real.algebra Real.dotModel p hbm domain.launch separate before work order
  refine ⟨after,run,?_,frame,Machine.closed p bases domain⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  have input : Real.Reads p (start (grid p) (n/p.bm,b*p.heads+h,0) before) read := loads
  have active : Source.active p (start (grid p) (n/p.bm,b*p.heads+h,0) before) (lane p hbm n d hd) := by
    change n/p.bm*p.bm+n%p.bm<p.length
    simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
  have answer := Real.answer_real p _ read input _ (domain.launch ⟨by omega,by omega,by omega⟩).bn active
  have expected := RealReturned.target_at_program p before read hbm b h n d hh hd
  rw [answer,expected]

theorem empty_kernel {α : Type} [Inhabited α] (model : Algebra α) (dot : Dot α)
    (p : Parameters) (hbm : 0<p.bm) (empty : ¬Live p) (before : State α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    work=[] ∧ runPrograms (fun _=>model) (fun _=>dot) p (grid p) work before=some before := by
  have nil : work=[] := List.nil_perm.mp (by simpa [RawKernel.no_grid p hbm empty] using order)
  exact ⟨nil,by simp [nil,runPrograms]⟩

#specNonCircular target avoiding [Physical.invoke,Whole.invoke,Whole.answer,FP.Structural.exec]
#axiomsClean kernel_correct
#axiomsClean empty_kernel
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelResults.Real

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelResults.Native
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine Whole002
noncomputable section

def reader (words : RegionName → Nat → BitVec 16) : RegionName → Nat → FamilyNative.Expr :=
  fun r off _=>.ready (.bits ((words r off).zeroExtend 32))

def kernel_reference (I : KernelOnly.Instructions) (layout : FamilyNative.Layout) (p : Parameters)
    (words : RegionName → Nat → BitVec 16) (b h n d : Nat) (hbm : 0<p.bm) (hd : d<p.dim) : BitVec 16 :=
  ((Native.lower (KernelOnly.Specification.algorithm (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
    p (reader words) (n/p.bm) b h (lane p hbm n d hd) (FamilyNative.physicalOwner layout d))).eval I.isa).setWidth 16

/-- Fixed native scalar/conversion/MMA graph, raw compiler routing and complete frame. -/
theorem kernel_native (I : KernelOnly.Instructions) (p : Parameters) (bases : Bases)
    (domain : Domain p bases) (hbm : 0<p.bm) (separate : Separated p)
    (numWarps numStages : Nat) (plan : RawKernel.Plan p numWarps numStages) (scaleWord : BitVec 32)
    (words : RegionName → Nat → BitVec 16) (before : State FamilyNative.Expr)
    (loads : ∀ r off,toFloat p.storage.float ((before.mem r off).read p.storage.tile)=reader words r off)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    let layout := plan.layout scaleWord
    ∃ after,runPrograms (fun _=>FamilyNative.algebra layout) (fun _=>FamilyNative.dotModel layout)
        p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ((Native.lower ((toFloat p.storage.float ((after.mem p.Out (logicalAddress p b h n d)).read p.storage.tile))
          (FamilyNative.physicalOwner layout d))).eval I.isa).setWidth 16=
          kernel_reference I layout p words b h n d hbm hd) ∧
      (∀ r off,Physical.Exterior p r off → after.mem r off=before.mem r off) ∧ Closed p bases := by
  dsimp only
  let layout := plan.layout scaleWord
  obtain ⟨after,run,values,frame⟩ := RawKernel.complete (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
    p hbm domain.launch separate before work order
  refine ⟨after,run,?_,frame,Machine.closed p bases domain⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd]
  simp only [Cell.read_same,Real.float_roundtrip]
  have positive : 0<p.heads := by omega
  have div : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh,Nat.zero_add]
  have mod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have algorithm := KernelOnly.source_algorithm (FamilyNative.algebra layout) (FamilyNative.dotModel layout) p
    (start (grid p) (n/p.bm,b*p.heads+h,0) before) (reader words) (n/p.bm) b h
    (by simp [start,ids]) (by simpa [start,ids] using div) (by simpa [start,ids] using mod) loads
  rw [algorithm]
  rfl

#axiomsClean kernel_native
#specNonCircular kernel_reference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelResults.Native
