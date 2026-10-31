import bench.tritonbench_g.matmul_triton2.Proofs.SourceIO
import bench.tritonbench_g.matmul_triton2.Proofs.SourceSafety
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

theorem launch_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (bounds : RegionName → Nat)
    (inBounds : (KernelOnly.IO.matmul rest P p).InBounds bounds xs)
    (scope : SourceBounds p)
    (pid : Nat) (active : pid < Geometry.count (Core.geometry p)) (s : State (Word α)) :
    (KernelOnly.IO.launch rest P p pid).Safe bounds s := by
  apply source_safe rest P p bounds _ scope active
  · intro q i hr hk
    have hk' : q*p.blockInner+i.2.1.val<p.innerSize := by omega
    have bound := inBounds.2.1 (0 : Fin 2)
      (⟨Core.row p ((KernelOnly.IO.launch rest P p pid).initial s) i.1,hr⟩,⟨_,hk'⟩,()) trivial
    change Core.row p ((KernelOnly.IO.launch rest P p pid).initial s) i.1*p.inputRow+
      (q*p.blockInner+i.2.1.val)*p.inputInner<bounds p.input at bound
    convert bound using 1
    unfold Core.aOffset
    ring
  · intro q i hk hc
    have hk' : q*p.blockInner+i.1.val<p.innerSize := by omega
    have bound := inBounds.2.1 (1 : Fin 2)
      (⟨_,hk'⟩,⟨Core.column p ((KernelOnly.IO.launch rest P p pid).initial s) i.2.1,hc⟩,()) trivial
    change (q*p.blockInner+i.1.val)*p.weightInner+
      Core.column p ((KernelOnly.IO.launch rest P p pid).initial s) i.2.1*p.weightColumn<bounds p.weight at bound
    convert bound using 1
    unfold Core.bOffset
    ring
  · intro i h
    exact inBounds.2.2.1 (0 : Fin 1)
      (⟨Core.row p ((KernelOnly.IO.launch rest P p pid).initial s) i.1,h.1⟩,
        ⟨Core.column p ((KernelOnly.IO.launch rest P p pid).initial s) i.2.1,h.2⟩,()) trivial

theorem schedule_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs p)) (bounds : RegionName → Nat)
    (inBounds : (KernelOnly.IO.matmul rest P p).InBounds bounds xs)
    (scope : SourceBounds p)
    (order : List Nat) (valid : ∀ pid ∈ order, pid < Geometry.count (Core.geometry p)) (s : State (Word α)) :
    ScheduleSafe bounds (order.map (KernelOnly.IO.launch rest P p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid order ih =>
    refine ⟨launch_safe rest P p xs bounds inBounds scope pid (valid pid (by simp)) s,?_⟩
    intro final _
    exact ih (fun next member => valid next (List.mem_cons.mpr (Or.inr member))) final

#axiomsClean launch_safe
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceIOProof
