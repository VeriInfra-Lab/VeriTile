import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOEvents
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

theorem mem_events (e : IO.Event order) : e ∈ IO.events order := by
  rcases e with ⟨launch,phase,lane⟩
  apply List.mem_flatMap.mpr
  refine ⟨launch,by simp,?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨phase,by simp,?_⟩
  exact List.mem_map.mpr ⟨lane,TileShape.mem_allIndices _ lane,rfl⟩

theorem last_sound (p : Fin 3 → Arguments) (order : List (Nat → Nat))
    (r : RegionName) (off : Nat) (e : IO.Event order) (h : IO.lastAt p order r off=some e) :
    IO.eventActive p e ∧ IO.eventRegion p e=r ∧ IO.eventOffset p e=off := by
  have:=List.find?_some h
  simpa using this

theorem last_none (p : Fin 3 → Arguments) (order : List (Nat → Nat))
    (r : RegionName) (off : Nat) :
    IO.lastAt p order r off=none ↔
      ∀ e : IO.Event order, ¬(IO.eventActive p e ∧ IO.eventRegion p e=r ∧ IO.eventOffset p e=off) := by
  simp only [IO.lastAt,List.find?_eq_none,List.mem_reverse,Bool.not_eq_true,decide_eq_false_iff_not]
  exact ⟨fun h e=>h e (mem_events e),fun h e _=>h e⟩

theorem covered (p : Fin 3 → Arguments) (e : IO.Event order) (active : IO.eventActive p e) :
    ∃ winner : IO.Event order, IO.Winner p winner ∧
      IO.eventRegion p winner=IO.eventRegion p e ∧ IO.eventOffset p winner=IO.eventOffset p e := by
  cases h : IO.lastAt p order (IO.eventRegion p e) (IO.eventOffset p e) with
  | none=>exact False.elim ((last_none p order _ _).mp h e ⟨active,rfl,rfl⟩)
  | some winner=>
    have sound:=last_sound p order _ _ winner h
    exact ⟨winner,by simpa only [IO.Winner,sound.2.1,sound.2.2] using h,sound.2⟩

theorem fold_select (items : List ι) (test : ι → Bool) (value : ι → C) (initial : C) :
    items.foldl (fun v i=>if test i then value i else v) initial=
      ((items.reverse.find? test).map value).getD initial := by
  induction items generalizing initial with
  | nil=>rfl
  | cons i rest ih=>
    simp only [List.foldl_cons,ih,List.reverse_cons,List.find?_append,List.find?_singleton]
    cases later : rest.reverse.find? test <;> cases hi : test i <;> simp

def value (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (vals : ArrayValues α (IO.inputs fp8 p)) (e : IO.Event order) : Cell α :=
  outputCell a (qkvOutputFP8 fp8 e.phase) (Target.raw a native fp8 p epsilon theta vals e)

def put (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (vals : ArrayValues α (IO.inputs fp8 p)) (s : State α) (e : IO.Event order) : State α :=
  if IO.eventActive p e then s.write (IO.eventRegion p e) (IO.eventOffset p e)
    (value a native fp8 p epsilon theta vals e) else s

theorem scatter_at (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (vals : ArrayValues α (IO.inputs fp8 p)) (items : List (IO.Event order))
    (s : State α) (r : RegionName) (off : Nat) :
    (items.foldl (put a native fp8 p epsilon theta vals) s).mem r off=
      items.foldl (fun v e=>if decide (IO.eventActive p e ∧ IO.eventRegion p e=r ∧ IO.eventOffset p e=off)
        then value a native fp8 p epsilon theta vals e else v) (s.mem r off) := by
  induction items generalizing s with
  | nil=>rfl
  | cons e rest ih=>
    rw [List.foldl_cons,ih,List.foldl_cons]
    congr 1
    by_cases active : IO.eventActive p e
    · rw [put,if_pos active]
      change (if r=IO.eventRegion p e ∧ off=IO.eventOffset p e then _ else _)=_
      simp only [active,true_and,decide_eq_true_eq]
      congr 1
      simp only [eq_comm]
    · simp only [put,if_neg active,active,false_and,decide_false,Bool.false_eq_true,if_false]

theorem scatter_last (a : Arithmetic α) (native fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (vals : ArrayValues α (IO.inputs fp8 p)) (s : State α) (r : RegionName) (off : Nat) :
    ((IO.events order).foldl (put a native fp8 p epsilon theta vals) s).mem r off=
      ((IO.lastAt p order r off).map (value a native fp8 p epsilon theta vals)).getD (s.mem r off) := by
  rw [scatter_at,fold_select]
  rfl

#axiomsClean covered
#axiomsClean scatter_last
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOEvents
