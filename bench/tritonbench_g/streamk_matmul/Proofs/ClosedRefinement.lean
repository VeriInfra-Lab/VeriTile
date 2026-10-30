import bench.tritonbench_g.streamk_matmul.Proofs.ClosedValues

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000

def extract (p : Parameters) (exterior : Nat) (s : Concurrent.State α) : Source.State α p exterior :=
  ⟨viewA p s,viewB p s,fun i=>s.mem 0 i.val,fun i=>s.mem 1 i.val,fun i=>s.mem 2 i.val⟩

inductive Corresponds (p : Parameters) (ops : Instructions α) :
    Source.Event Concurrent.Expr p → Source.Event α p → Prop where
  | output (i : C p) (source : (A p → Concurrent.Expr) → (B p → Concurrent.Expr) → Concurrent.Expr)
      (target : (A p → α) → (B p → α) → α)
      (value : ∀ s,(source (inputA p) (inputB p)).eval (backend ops) s=target (viewA p s) (viewB p s)) :
      Corresponds p ops (.storeOutput i source) (.storeOutput i target)
  | scratch (i : S p) (source : (A p → Concurrent.Expr) → (B p → Concurrent.Expr) → Concurrent.Expr)
      (target : (A p → α) → (B p → α) → α)
      (value : ∀ s,(source (inputA p) (inputB p)).eval (backend ops) s=target (viewA p s) (viewB p s)) :
      Corresponds p ops (.storeScratch i source) (.storeScratch i target)
  | atomic (i : C p) (j : S p) : Corresponds p ops (.atomic i j) (.atomic i j)

theorem state_ext {a b : Source.State α p exterior}
    (left : a.inputA=b.inputA) (right : a.inputB=b.inputB) (output : a.output=b.output)
    (scratch : a.scratch=b.scratch) (outside : a.exterior=b.exterior) : a=b := by
  cases a
  cases b
  simp_all

theorem step_projection (p : Parameters) (ops : Instructions α) (exterior : Nat)
    (symbolicEvent : Source.Event Concurrent.Expr p) (actualEvent : Source.Event α p)
    (related : Corresponds p ops symbolicEvent actualEvent) (s : Concurrent.State α) :
    extract p exterior (Concurrent.execute (backend ops) (lowerEvent p symbolicEvent) s)=
      Memory.step ops.atomicAdd actualEvent (extract p exterior s) := by
  cases related with
  | output i source target value =>
    apply state_ext
    · rfl
    · rfl
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step,
        Function.update_apply,value s,Fin.ext_iff]
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]
  | scratch i source target value =>
    apply state_ext
    · rfl
    · rfl
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step,
        Function.update_apply,value s,Fin.ext_iff]
    · funext j
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]
  | atomic i j =>
    apply state_ext
    · rfl
    · rfl
    · funext k
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step,
        Function.update_apply,backend,Concurrent.Expr.eval,Fin.ext_iff]
    · funext k
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]
    · funext k
      simp [extract,Concurrent.execute,lowerEvent,Concurrent.update,Memory.step]

theorem run_projection (p : Parameters) (ops : Instructions α) (exterior : Nat)
    (symbolicEvents : List (Source.Event Concurrent.Expr p)) (actualEvents : List (Source.Event α p))
    (related : List.Forall₂ (Corresponds p ops) symbolicEvents actualEvents) (s : Concurrent.State α) :
    extract p exterior (Concurrent.run (backend ops) (symbolicEvents.map (lowerEvent p)) s)=
      Memory.run ops.atomicAdd actualEvents (extract p exterior s) := by
  induction related generalizing s with
  | nil=>rfl
  | @cons first second rest tail headRelated tailRelated ih=>
    simp only [List.map_cons,Concurrent.run,Memory.run,List.foldl_cons]
    rw [ih,step_projection p ops exterior first second headRelated s]
    rfl

theorem forall₂_flatMap {R : α → β → Prop} (xs : List ι) (f : ι → List α) (g : ι → List β)
    (related : ∀ i,List.Forall₂ R (f i) (g i)) : List.Forall₂ R (xs.flatMap f) (xs.flatMap g) := by
  induction xs with
  | nil=>exact .nil
  | cons i rest ih=>exact List.rel_append (related i) ih

theorem forall₂_map {R : α → β → Prop} (xs : List ι) (f : ι → α) (g : ι → β)
    (related : ∀ i,R (f i) (g i)) : List.Forall₂ R (xs.map f) (xs.map g) := by
  induction xs with
  | nil=>exact .nil
  | cons i rest ih=>exact .cons (related i) ih

theorem storeTile_corresponds (p : Parameters) (ops : Instructions α) (tile : Nat)
    (symbolicValue : Lane p → (A p → Concurrent.Expr) → (B p → Concurrent.Expr) → Concurrent.Expr)
    (actualValue : Lane p → (A p → α) → (B p → α) → α)
    (value : ∀ lane s,(symbolicValue lane (inputA p) (inputB p)).eval (backend ops) s=
      actualValue lane (viewA p s) (viewB p s)) :
    List.Forall₂ (Corresponds p ops) (storeTile p symbolicValue tile) (storeTile p actualValue tile) := by
  unfold storeTile
  apply forall₂_flatMap
  intro lane
  by_cases active : Source.active p tile lane
  · simp only [dif_pos active]
    exact .cons (.output _ _ _ (value lane)) .nil
  · simp only [dif_neg active]
    exact .nil

theorem mac_corresponds (p : Parameters) (ops : Instructions α) (program : Fin p.h.programs)
    (c : Protocol.Chunk) :
    List.Forall₂ (Corresponds p ops) (macLoop p symbolic program c) (macLoop p ops program c) := by
  unfold macLoop
  dsimp only
  split
  · apply storeTile_corresponds
    intro lane s
    exact chunk_eval p ops s _ _ _ lane
  · apply forall₂_map
    intro lane
    exact .scratch _ _ _ (fun s=>chunk_eval p ops s _ _ _ lane)

theorem first_corresponds (p : Parameters) (ops : Instructions α) :
    List.Forall₂ (Corresponds p ops) (firstWave p symbolic) (firstWave p ops) := by
  unfold firstWave
  split
  · apply forall₂_flatMap
    intro program
    apply forall₂_flatMap
    intro c
    exact mac_corresponds p ops program c
  · exact .nil

theorem reduce_corresponds (p : Parameters) (ops : Instructions α) :
    List.Forall₂ (Corresponds p ops) (reduceWave p) (reduceWave p) := by
  apply forall₂_map
  intro pair
  exact .atomic pair.1 pair.2

theorem full_corresponds (p : Parameters) (ops : Instructions α) :
    List.Forall₂ (Corresponds p ops) (fullWave p symbolic) (fullWave p ops) := by
  unfold fullWave
  apply forall₂_flatMap
  intro pid
  apply storeTile_corresponds
  intro lane s
  change ops.halfRoundTrip _=ops.halfRoundTrip _
  rw [chunk_eval]

#axiomsClean step_projection
#axiomsClean run_projection
#axiomsClean first_corresponds
#axiomsClean reduce_corresponds
#axiomsClean full_corresponds
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
