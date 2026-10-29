import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOSafety
import bench.repairs.tritonbench_g.matmul_triton1.IO

namespace VeriTile.Bench.Repairs.MatmulTriton1.IOGridSafety
open VeriTile.Triton FP.Structural Revision001 MatmulIO
noncomputable section
variable {α : Type} [Inhabited α]

theorem launch_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (bounds : RegionName → Nat)
    (hb : (io rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
    (pid : Nat) (s : State α) : (launch rest P p pid).Safe bounds s := by
  apply IOSafety.source_safe rest P p bounds _ hBK
  · intro q i hr hk
    have hk' : q*p.blockInner+i.2.1.val<p.innerSize := by
      change i.2.1.val<p.innerSize-q*p.blockInner at hk
      omega
    have bound := hb.2.1 (0 : Fin 2)
      (⟨Core.row p ((launch rest P p pid).initial s) i.1,hr⟩,⟨_,hk'⟩,()) trivial
    change Core.row p ((launch rest P p pid).initial s) i.1*p.inputRow+
      (q*p.blockInner+i.2.1.val)*p.inputInner<bounds p.input at bound
    convert bound using 1
    unfold Core.aOffset
    ring
  · intro q i hk hc
    have hk' : q*p.blockInner+i.1.val<p.innerSize := by
      change i.1.val<p.innerSize-q*p.blockInner at hk
      omega
    have bound := hb.2.1 (1 : Fin 2)
      (⟨_,hk'⟩,⟨Core.column p ((launch rest P p pid).initial s) i.2.1,hc⟩,()) trivial
    change (q*p.blockInner+i.1.val)*p.weightInner+
      Core.column p ((launch rest P p pid).initial s) i.2.1*p.weightColumn<bounds p.weight at bound
    convert bound using 1
    unfold Core.bOffset
    ring
  · intro i h
    exact hb.2.2.1 (0 : Fin 1)
      (⟨Core.row p ((launch rest P p pid).initial s) i.1,h.1⟩,
        ⟨Core.column p ((launch rest P p pid).initial s) i.2.1,h.2⟩,()) trivial

theorem schedule_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (bounds : RegionName → Nat)
    (hb : (io rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
    (order : List Nat) (s : State α) :
    ScheduleSafe bounds (order.map (launch rest P p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid order ih =>
    exact ⟨launch_safe rest P p xs bounds hb hBK pid s,fun final _ => ih final⟩

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOGridSafety
