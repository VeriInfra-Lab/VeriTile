import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelExecution
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelRealReference

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReal
open VeriTile.Triton FP.Structural KernelProgram KernelRealReference
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def expected (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (u : State ℝ) (r : RegionName) (t : Nat) (i : Lane) : ℝ :=
  let a := inputs (KernelReference.read c s p u)
  match kind with
  | .forward => Formula.forward a seed t i.1 i.2.1
  | .backward =>
    if r="DI" then Formula.reverse a c.N (t+1) i.1 i.2.1
    else if r="DG" then ∑ k : Fin 1024,
      Formula.reverse a c.N (t+1) (index k).1 (index k).2.1 *
        (KernelReference.read c s p u).saved t (index k)
    else Formula.reverse a c.N 0 i.1 i.2.1
def stored (kind : Kind) (r : RegionName) (value : ℝ) : Cell ℝ :=
  .mk (KernelExecution.dtype kind r).toTileDType (ofFloat (KernelExecution.dtype kind r) value)
def Outputs (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c)
    (u v : State ℝ) : Prop := match kind with
  | .forward => ∀ (t : Fin c.N) (e : Fin 64) (j : Fin 16),
      v.mem "O" (address c p t.val e.val j.val)=
        stored .forward "O" (expected .forward c s seed p u "O" t.val (e,j,PUnit.unit))
  | .backward =>
    (∀ (t : Fin (c.N-1)) (e : Fin 64) (j : Fin 16),
      v.mem "DI" (address c p t.val e.val j.val)=
        stored .backward "DI" (expected .backward c s seed p u "DI" t.val (e,j,PUnit.unit))) ∧
    (∀ t : Fin (c.N-1),v.mem "DG" (gradientAddress c p t.val)=
      stored .backward "DG" (expected .backward c s seed p u "DG" t.val (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit))) ∧
    (∀ (e : Fin 64) (j : Fin 16),v.mem "DL" (dseedAddress c p e.val j.val)=
      stored .backward "DL" (expected .backward c s seed p u "DL" 0 (e,j,PUnit.unit)))

theorem reverse_value (c : Shape) (a : KernelReference.Inputs ℝ) (t : Fin (c.N-1)) :
    KernelReference.reverse algebra a c.N (c.N-1-t.val)=
      (fun i => Formula.reverse (inputs a) c.N (t.val+1) i.1 i.2.1) := by
  funext ⟨e,j,⟨⟩⟩
  rw [reverse_formula a c.N _ (by omega)]
  congr 1
  omega

theorem complete (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State ℝ) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v,KernelExecution.Facts algebra kind c s seed p u v ∧ Outputs kind c s seed p u v := by
  obtain ⟨v,facts⟩ := KernelExecution.complete algebra kind c s seed p u nonempty hh hk hv
  refine ⟨v,facts,?_⟩
  cases kind with
  | forward =>
    intro t e j
    rw [facts.outputs t e j]
    change KernelReference.stored algebra .real
      (KernelReference.forward algebra (KernelReference.read c s p u) seed t.val (e,j,PUnit.unit))=_
    rw [KernelRealReference.forward_formula]
    rfl
  | backward =>
    refine ⟨?_,?_,?_⟩
    · intro t e j
      rw [facts.outputs.1 t e j]
      simp only [KernelExecution.expected,↓reduceIte,reverse_value c _ t]
      rfl
    · intro t
      rw [facts.outputs.2.1 t]
      simp only [KernelExecution.expected,show ("DG" : RegionName)≠"DI" by decide,
        if_false,if_true,reverse_value c _ t,contraction_formula]
      rfl
    · intro e j
      rw [facts.outputs.2.2 e j]
      simp only [KernelExecution.expected,show ("DL" : RegionName)≠"DI" by decide,
        show ("DL" : RegionName)≠"DG" by decide,if_false,reverse_formula _ c.N c.N (Nat.le_refl _),Nat.sub_self]
      rfl

#axiomsClean complete
#specNonCircular expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReal
