import bench.tritonbench_g.chunk_delta_fwd.Proofs.Grid
import bench.tritonbench_g.chunk_delta_fwd.Proofs.Ownership
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open Geometry
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def historyValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (input : Memory α) (head c key value : Nat) (hk : key < p.K) :=
  cast M (states M D p (Ownership.pid p head value) input c (Ownership.hLane p h key value hk))
def correctedValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (input : Memory α) (head t value : Nat) :=
  cast M (corrected M D p (Ownership.pid p head value) input (t/p.BT) (Ownership.block p t)
    (states M D p (Ownership.pid p head value) input (t/p.BT)) (Ownership.cLane p h t value))
def finalValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (input : Memory α) (head key value : Nat) (hk : key < p.K) :=
  historyValue M D p h input head p.NT key value hk

def OutputCell (p : Parameters) (r : RegionName) (o : Nat) : Prop :=
  (r="history" ∧ o < p.B*p.H*p.NT*p.K*p.V) ∨
  (r="corrected" ∧ o < p.B*p.H*p.T*p.V) ∨
  (r="final" ∧ p.storeFinal=Bool.true ∧ o < p.B*p.H*p.K*p.V)

structure GridSpec (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (input before after : Memory α) : Prop where
  history : ∀ head, head < p.B*p.H → ∀ c, c < p.NT → ∀ key (hk : key < p.K),
    ∀ value, value < p.V → after "history" (((head*p.NT+c)*p.K+key)*p.V+value)=
      .mk .real (historyValue M D p h input head c key value hk)
  corrected : ∀ head, head < p.B*p.H → ∀ t, t < p.T → ∀ value, value < p.V →
    after "corrected" ((head*p.T+t)*p.V+value)=.mk .real (correctedValue M D p h input head t value)
  final : p.storeFinal=Bool.true → ∀ head, head < p.B*p.H → ∀ key (hk : key < p.K),
    ∀ value, value < p.V → after "final" ((head*p.K+key)*p.V+value)=
      .mk .real (finalValue M D p h input head key value hk)
  frame : ∀ r o, ¬OutputCell p r o → after r o=before r o

theorem footprint_allocated (p : Parameters) (pid : PID) (hp : ValidPID p pid)
    (r : RegionName) (o : Nat) (hw : Footprint p pid r o) : OutputCell p r o := by
  rcases hw with ⟨rfl,c,hc,i,hi,rfl⟩|⟨rfl,c,hc,b,hb,i,hi,rfl⟩|⟨rfl,hf,i,hi,rfl⟩
  · exact Or.inl ⟨rfl,history_bound p pid c i hp hc hi⟩
  · exact Or.inr (Or.inl ⟨rfl,corrected_bound p pid c b i hp hi⟩)
  · exact Or.inr (Or.inr ⟨rfl,hf,final_bound p pid i hp hi⟩)

theorem grid_spec (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (schedule : List PID) (hs : schedule.Perm (grid p)) (input m : Memory α) :
    GridSpec M D p h input m (gridMemory M D p schedule input m) := by
  have member : ∀ pid, ValidPID p pid → pid∈schedule := fun pid hp =>
    hs.mem_iff.mpr ((mem_grid p pid).mpr hp)
  constructor
  · intro head hh c hc key hk value hv
    let pid := Ownership.pid p head value
    let i := Ownership.hLane p h key value hk
    have hi := Ownership.h_active p h head key value hk hv
    have hmem := member pid (Ownership.valid p h head value hh hv)
    rw [← Ownership.h_offset p h head c key value hk]
    exact grid_read M D p schedule input m pid "history" (hOffset p pid c i) _ hmem
      (Or.inl ⟨rfl,c,hc,i,hi,rfl⟩) (fun mem => program_history M D p pid input mem c i hc hi)
  · intro head hh t ht value hv
    let pid := Ownership.pid p head value
    let i := Ownership.cLane p h t value
    have hi := Ownership.c_active p h head t value ht hv
    have hc := Ownership.chunk_lt p h t ht
    have hb := Ownership.block_lt p h t
    have hmem := member pid (Ownership.valid p h head value hh hv)
    rw [← Ownership.c_offset p h head t value]
    exact grid_read M D p schedule input m pid "corrected" (cOffset p pid (t/p.BT) (Ownership.block p t) i)
      _ hmem (Or.inr (Or.inl ⟨rfl,t/p.BT,hc,Ownership.block p t,hb,i,hi,rfl⟩))
      (fun mem => program_corrected M D p pid input mem (t/p.BT) (Ownership.block p t) i hc hb hi
        (Ownership.base_span p h t value))
  · intro hp head hh key hk value hv
    let pid := Ownership.pid p head value
    let i := Ownership.hLane p h key value hk
    have hi := Ownership.h_active p h head key value hk hv
    have hmem := member pid (Ownership.valid p h head value hh hv)
    rw [← Ownership.f_offset p h head key value hk]
    exact grid_read M D p schedule input m pid "final" (fOffset p pid i) _ hmem
      (Or.inr (Or.inr ⟨rfl,hp,i,hi,rfl⟩)) (fun mem => program_final M D p pid input mem i hp hi)
  · intro r o ho
    apply grid_frame
    intro pid hpid hw
    exact ho (footprint_allocated p pid ((mem_grid p pid).mp (hs.mem_iff.mp hpid)) r o hw)

theorem whole_grid (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (h : ShapeValid p)
    (schedule : List PID) (hs : schedule.Perm (grid p)) (input m : Memory α) (hi : SameInputs m input) :
    ∃ after, runGrid M D p schedule m=some after ∧ GridSpec M D p h input m after :=
  ⟨gridMemory M D p schedule input m,(grid_execution M D p schedule input m hi).1,
    grid_spec M D p h schedule hs input m⟩

#axiomsClean grid_spec
#axiomsClean whole_grid
#specNonCircular GridSpec avoiding [run,FP.Structural.exec,runGrid,gridMemory]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
