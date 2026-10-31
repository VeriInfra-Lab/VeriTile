import bench.tritonbench_g.matmul_triton2.Proofs.SourceGrid
import bench.tritonbench_g.matmul_triton2.Proofs.SourceObservation
import bench.tritonbench_g.matmul_triton2.IO
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceIOProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Bench.Repairs.MatmulTriton2
open Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem launch_initial (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (pid : Nat) (s : State (Word α)) :
    (KernelOnly.IO.launch rest P p pid).initial s = GridExecution.atProgram s pid := rfl

theorem schedule_eq (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (order : List Nat) (s : State (Word α)) :
    runSchedule (order.map (KernelOnly.IO.launch rest P p)) s = GridExecution.sequence (one rest P p) order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid restOrder ih =>
    simp only [List.map_cons,runSchedule,GridExecution.sequence]
    have tailEq := funext ih
    rw [tailEq]
    rfl

theorem read_left (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (s : State (Word α))
    (hin : (KernelOnly.IO.matmul rest P p).InputBinding xs s) (r : Fin p.rows) (k : Fin p.innerSize) :
    ((observedState P s).mem p.input (r.val*p.inputRow+k.val*p.inputInner)).read .real =
      xs (0 : Fin 2) (r,k,()) := by
  change (mapCell (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P) (s.mem _ _)).read .real = _
  rw [observed_cell_read]
  have h := hin (0 : Fin 2) (r,k,()) trivial
  change (s.mem p.input (r.val*p.inputRow+k.val*p.inputInner)).read .real =
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value.input (xs (0 : Fin 2) (r,k,())) at h
  rw [h]
  rfl

theorem read_right (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (s : State (Word α))
    (hin : (KernelOnly.IO.matmul rest P p).InputBinding xs s) (k : Fin p.innerSize) (c : Fin p.columns) :
    ((observedState P s).mem p.weight (k.val*p.weightInner+c.val*p.weightColumn)).read .real =
      xs (1 : Fin 2) (k,c,()) := by
  change (mapCell (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P) (s.mem _ _)).read .real = _
  rw [observed_cell_read]
  have h := hin (1 : Fin 2) (k,c,()) trivial
  change (s.mem p.weight (k.val*p.weightInner+c.val*p.weightColumn)).read .real =
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value.input (xs (1 : Fin 2) (k,c,())) at h
  rw [h]
  rfl

theorem precision_target (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (s : State (Word α))
    (hin : (KernelOnly.IO.matmul rest P p).InputBinding xs s) (r : Fin p.rows) (c : Fin p.columns) (n : Nat) :
    Core.logicalAccum P p (observedState P s) r.val c.val n = VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionAccum P p xs r c n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Core.logicalAccum,VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionAccum,ih]
    congr 1
    · funext k
      have mask : k.val<Core.inner p-n*p.blockInner ↔ n*p.blockInner+k.val<p.innerSize := by unfold Core.inner; omega
      simp only [Core.logicalA,VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left,mask]
      split <;> rename_i h
      · exact read_left rest P p xs s hin r ⟨_,h⟩
      · rfl
    · funext k
      have mask : k.val<Core.inner p-n*p.blockInner ↔ n*p.blockInner+k.val<p.innerSize := by unfold Core.inner; omega
      simp only [Core.logicalB,VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right,mask]
      split <;> rename_i h
      · exact read_right rest P p xs s hin ⟨_,h⟩ c
      · rfl

#axiomsClean launch_initial
#axiomsClean schedule_eq
#axiomsClean read_left
#axiomsClean read_right
#axiomsClean precision_target
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceIOProof
