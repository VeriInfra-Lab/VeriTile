import bench.tritonbench_g.matmul_triton2.Proofs.MaskIOComplete
import bench.tritonbench_g.matmul_triton2.Proofs.MaskIOSafety
import bench.tritonbench_g.matmul_triton2.Proofs.Real
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.MaskIOProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
open VeriTile.Bench.Repairs.MatmulTriton2
open Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem precision {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (scope : SourceBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (KernelOnly.Pair.candidateIO rest P p).Implements (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionTarget P p) := by
  apply ProgramIO.Implements.intro
  · intro vals limits bound initial _
    exact schedule_safe rest P p vals limits bound scope _
      (fun _ member => List.mem_range.mp member) initial
  · exact complete rest P p scope geometry ha hb

private theorem real_sum_same (f : Nat → ℝ) (n : Nat) :
    RealSum.sum f n=VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum f n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [RealSum.sum,VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum,ih]

theorem real_target (rest : Algebra ℝ) (p : Core.Parameters)
    (xs : ChannelValues ℝ (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (s : State (Word ℝ))
    (hin : (KernelOnly.Pair.candidateIO rest Core.realPrimitives p).InputBinding xs s)
    (r : Fin p.rows) (c : Fin p.columns) :
    Core.realMatmul p (observedState Core.realPrimitives s) r.val c.val =
      (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.realTarget p xs) (0 : Fin 1) (r,c,()) := by
  change RealSum.sum (Core.realProduct p (observedState Core.realPrimitives s) r.val c.val) p.innerSize =
    VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum
      (fun k => VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left p xs 0 r k *
        VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right p xs 0 c k) p.innerSize
  rw [← real_sum_same]
  apply RealSum.sum_congr
  intro k hk
  change k<p.innerSize at hk
  change ((observedState Core.realPrimitives s).mem p.input (r.val*p.inputRow+k*p.inputInner)).read .real *
    ((observedState Core.realPrimitives s).mem p.weight (k*p.weightInner+c.val*p.weightColumn)).read .real =
    VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left p xs 0 r k * VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right p xs 0 c k
  rw [read_left rest Core.realPrimitives p xs s hin r ⟨k,hk⟩,
    read_right rest Core.realPrimitives p xs s hin ⟨k,hk⟩ c]
  simp [VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left,VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right,hk]

theorem real (rest : Algebra ℝ) (p : Core.Parameters) (scope : SourceBounds p) (inner : 0<p.blockInner)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (KernelOnly.Pair.candidateIO rest Core.realPrimitives p).Implements (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.realTarget p) := by
  apply ProgramIO.Implements.intro
  · intro vals limits bound initial _
    exact schedule_safe rest Core.realPrimitives p vals limits bound scope _
      (fun _ member => List.mem_range.mp member) initial
  · intro xs initial hin
    obtain ⟨final,ran,outputs,frame⟩ :=
      complete rest Core.realPrimitives p scope geometry ha hb xs initial hin
    have targets : VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionTarget Core.realPrimitives p xs=VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.realTarget p xs := by
      funext o j
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      change VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionAccum Core.realPrimitives p xs j.1 j.2.1 (Core.steps p)=_
      rw [← precision_target rest Core.realPrimitives p xs initial hin]
      exact (Core.logical_realMatmul p (observedState Core.realPrimitives initial) _ _
        inner).trans
        (real_target rest p xs initial hin j.1 j.2.1)
    rw [targets] at outputs
    exact ⟨final,ran,outputs,frame⟩

#axiomsClean precision
#axiomsClean real_target
#axiomsClean real
end
end VeriTile.Bench.Repairs.MatmulTriton2.MaskIOProof
