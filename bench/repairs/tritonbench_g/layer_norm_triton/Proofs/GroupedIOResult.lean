import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.GroupedFoldIO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedSafety

namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedIOResult
open VeriTile.Triton FP.Structural SourceModel Interface SourceGroupedGraph SourceGroupedExecution SourceGroupedResult
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem implements (m : Model α) (stride M N G B group : Nat)
    (observe : RegionName → Nat → α → α) :
    (Grouped.io m stride M N G B group observe).Implements
      (GroupedReference.expected m stride M N G B group observe) := by
  intro vals _ bounds boundsOK initial binding
  have safe : ScheduleSafe bounds ((Grouped.io m stride M N G B group observe).schedule vals) initial := by
    change ScheduleSafe bounds [job m
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) group] initial
    rw [scheduleSafe_singleton]
    apply SourceSafety.Grouped.source_safe m
    · intro row hr j hj
      exact boundsOK.1.2.1 (0 : Fin 6) (⟨row,hr⟩,j,()) hj
    · intro row hr j hj
      exact boundsOK.1.2.1 (1 : Fin 6) (⟨row,hr⟩,j,()) hj
    · intro j hj
      exact boundsOK.1.2.1 (2 : Fin 6) (j,()) hj
    · intro row hr
      exact boundsOK.1.2.1 (3 : Fin 6) (⟨row,hr⟩,()) trivial
    · intro row hr
      exact boundsOK.1.2.1 (4 : Fin 6) (⟨row,hr⟩,()) trivial
    · intro row hr j hj
      exact boundsOK.1.2.2.1 (0 : Fin 3) (⟨row,hr⟩,⟨j.val,hj⟩,()) trivial
    · intro j hj
      exact boundsOK.1.2.2.1 (1 : Fin 3) (j,()) hj
    · intro j hj
      exact boundsOK.1.2.2.1 (2 : Fin 3) (j,()) hj
  let start := (job m
    (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) group).initial initial
  have inputs : GroupedValues.Inputs start stride M N B
      (Grouped.x vals) (Grouped.dy vals) (Grouped.w vals) (Grouped.mean vals) (Grouped.inverse vals) := by
    constructor
    · intro row hr j hj
      simpa [start,Invocation.initial,bindRegisters,Grouped.io,Grouped.layout,Grouped.inputs,
        Grouped.x,hr,mapValue] using binding (0 : Fin 6) (⟨row,hr⟩,j,()) hj
    · intro row hr j hj
      simpa [start,Invocation.initial,bindRegisters,Grouped.io,Grouped.layout,Grouped.inputs,
        Grouped.dy,hr,mapValue] using binding (1 : Fin 6) (⟨row,hr⟩,j,()) hj
    · intro j hj
      exact binding (2 : Fin 6) (j,()) hj
    · intro row hr
      simpa [start,Invocation.initial,bindRegisters,Grouped.io,Grouped.layout,Grouped.inputs,
        Grouped.mean,hr,mapValue] using binding (3 : Fin 6) (⟨row,hr⟩,()) trivial
    · intro row hr
      simpa [start,Invocation.initial,bindRegisters,Grouped.io,Grouped.layout,Grouped.inputs,
        Grouped.inverse,hr,mapValue] using binding (4 : Fin 6) (⟨row,hr⟩,()) trivial
  obtain ⟨t,ran,memory,partials,frame⟩ := SourceGroupedResult.executes m stride M N G B start
  have fold (query : Nat) := GroupedValues.fold_value m start stride M N G B group group query
    _ _ _ _ _ inputs (zero m B)
  refine ⟨safe,t,(runSchedule_singleton _ _).trans ran,?_,?_⟩
  · intro o lane active
    fin_cases o
    · have val := congrArg Prod.fst (fold (lane.1.val*stride+lane.2.1.val))
      have prior := binding (5 : Fin 6) lane trivial
      change (start.mem "DX" (lane.1.val*stride+lane.2.1.val)).read .real=
        Grouped.priorDX vals lane.1 lane.2.1 at prior
      change observe "DX" (lane.1.val*stride+lane.2.1.val)
        ((t.mem "DX" (lane.1.val*stride+lane.2.1.val)).read .real)=_
      rw [memory]
      unfold SourceGroupedExecution.final
      rw [stores_frame _ N B _ _ "DX" _ (.inl ⟨by decide,by decide⟩)]
      change observe "DX" (lane.1.val*stride+lane.2.1.val)
        (((SourceGroupedExecution.fold m stride M N G B group group start (zero m B)).1.mem
          "DX" (lane.1.val*stride+lane.2.1.val)).read .real)=_
      dsimp only [Prod.fst] at val
      rw [val,prior]
      rfl
    · have val := congrArg Prod.snd (fold 0)
      have independent := GroupedValues.carries_independent m stride M N G B group group 0 0
        (Grouped.x vals) (Grouped.dy vals) (Grouped.w vals) (Grouped.mean vals) (Grouped.inverse vals)
        ((start.mem "DX" 0).read .real) default (GroupedValues.flat (zero m B))
      have eq := congrArg (fun a : GroupedReference.Carries α B => a.1 lane.1) (val.trans independent)
      change observe "DW" (group*N+lane.1.val) ((t.mem "DW" (group*N+lane.1.val)).read .real)=_
      have hp := (partials lane active).1
      change t.mem "DW" (group*N+lane.1.val)=_ at hp
      rw [hp,Cell.read_same]
      change observe "DW" (group*N+lane.1.val)
        ((GroupedValues.flat (SourceGroupedExecution.fold m stride M N G B group group start (zero m B)).2).1 lane.1)=_
      dsimp only [Prod.snd] at eq
      rw [eq]
      rfl
    · have val := congrArg Prod.snd (fold 0)
      have independent := GroupedValues.carries_independent m stride M N G B group group 0 0
        (Grouped.x vals) (Grouped.dy vals) (Grouped.w vals) (Grouped.mean vals) (Grouped.inverse vals)
        ((start.mem "DX" 0).read .real) default (GroupedValues.flat (zero m B))
      have eq := congrArg (fun a : GroupedReference.Carries α B => a.2 lane.1) (val.trans independent)
      change observe "DB" (group*N+lane.1.val) ((t.mem "DB" (group*N+lane.1.val)).read .real)=_
      have hp := (partials lane active).2
      change t.mem "DB" (group*N+lane.1.val)=_ at hp
      rw [hp,Cell.read_same]
      change observe "DB" (group*N+lane.1.val)
        ((GroupedValues.flat (SourceGroupedExecution.fold m stride M N G B group group start (zero m B)).2).2 lane.1)=_
      dsimp only [Prod.snd] at eq
      rw [eq]
      rfl
  · intro r off outside
    rw [memory]
    unfold SourceGroupedExecution.final SourceGroupedExecution.stores
    rw [State.masked_scatter_frame,State.masked_scatter_frame]
    · apply SourceGroupedResult.fold_frame
      by_cases hr : r≠"DX"
      · exact .inl hr
      · right
        intro row hrow j hj
        exact (outside.1 (0 : Fin 3) (⟨row,hrow⟩,⟨j,hj⟩,()) trivial).resolve_left hr
    · by_cases hw : r≠"DW"
      · exact .inl hw
      · right
        intro j _ hj
        exact (outside.1 (1 : Fin 3) j hj).resolve_left hw
    · by_cases hb : r≠"DB"
      · exact .inl hb
      · right
        intro j _ hj
        exact (outside.1 (2 : Fin 3) j hj).resolve_left hb
#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedIOResult
