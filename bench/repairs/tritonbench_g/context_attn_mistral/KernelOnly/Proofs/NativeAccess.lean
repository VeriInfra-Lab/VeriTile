import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.NativeProgramSafety
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.NativeIOBindings

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

theorem tile_bounds (c : Host.Call) (idx : GridIndex c.grid) (s : State α)
    (metadata : KernelFootprint.Metadata c s) (bounds : RegionName → Nat)
    (reads : ∀ port off,off∈NativeArrays.tileReads c idx (NativeBindings.snapshot c s) port →
      off<bounds (NativeArrays.ports c)[port].buf)
    (writes : ∀ i,c.active idx i → c.offset idx i<bounds c.regions.out) :
    NativeSourceSafety.Bounds c.layout c.regions (idx.toPids 0)
      (KernelReference.inputs c.layout c.regions (KernelReference.atIndex s idx)) bounds := by
  rw [←NativeBindings.tileInput_snapshot]
  let a:=NativeArrays.tileInput c idx (NativeBindings.snapshot c s)
  constructor
  · exact reads 3 _ (by change _∈[idx.toPids 0]; simp)
  · exact reads 4 _ (by change _∈[idx.toPids 0]; simp)
  · intro i d on
    apply reads 0
    apply List.mem_filterMap.mpr
    refine ⟨(i,d,()),TileShape.mem_allIndices _ _,?_⟩
    simp only [if_pos on]
  · intro cur nonnegative active d j on
    apply reads 1
    apply List.mem_flatMap.mpr
    refine ⟨cur.toNat,?_,?_⟩
    · apply List.mem_range.mpr
      omega
    · have curNat : (cur.toNat : Int)=cur:=Int.toNat_of_nonneg nonnegative
      apply List.mem_filterMap.mpr
      refine ⟨(d,j,()),TileShape.mem_allIndices _ _,?_⟩
      simp only [curNat,if_pos on]
  · intro cur nonnegative active j d on
    apply reads 2
    apply List.mem_flatMap.mpr
    refine ⟨cur.toNat,?_,?_⟩
    · apply List.mem_range.mpr
      omega
    · have curNat : (cur.toNat : Int)=cur:=Int.toNat_of_nonneg nonnegative
      apply List.mem_filterMap.mpr
      refine ⟨(j,d,()),TileShape.mem_allIndices _ _,?_⟩
      simp only [curNat,if_pos on]
  · intro i d on
    have active : c.active idx (i,d,()) :=
      (KernelFootprint.active_matches c s metadata idx (i,d,())).mp on
    have output:=writes (i,d,()) active
    rwa [←KernelFootprint.offset_matches c s metadata idx (i,d,())] at output

theorem schedule_safe (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (metadata : KernelFootprint.Metadata c s)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (bounds : RegionName → Nat)
    (reads : ∀ port off,off∈NativeArrays.reads c o jobs (NativeBindings.snapshot c s) port →
      off<bounds (NativeArrays.ports c)[port].buf)
    (writes : ∀ idx∈jobs,∀ i,c.active idx i → c.offset idx i<bounds c.regions.out) :
    ScheduleSafe bounds (jobs.map (invocation c o)) s := by
  induction jobs generalizing s with
  | nil=>trivial
  | cons idx rest ih=>
    constructor
    · apply NativeSourceSafety.complete c.layout c.regions (Host.scale c.layout.D) o
        (by rw [c.blockN];decide) (KernelReference.atIndex s idx) bounds
      apply tile_bounds c idx s metadata bounds
      · intro port off member
        exact reads port off (List.mem_append_left _ member)
      · exact writes idx (List.mem_cons_self ..)
    · intro t executed
      obtain ⟨u,ran,memory⟩:=KernelReference.exec_complete c.layout c.regions (Host.scale c.layout.D) o
        (by rw [c.blockN];decide) (KernelReference.atIndex s idx)
      have same:t=u:=Option.some.inj (executed.symm.trans ran)
      subst t
      change u.mem=(KernelReference.tile c o idx s).mem at memory
      have snap : NativeBindings.snapshot c u=NativeArrays.tile c o idx (NativeBindings.snapshot c s) :=
        (snapshot_congr c _ _ memory).trans (NativeBindings.tile_snapshot c o idx s metadata)
      have metadataState : KernelFootprint.Metadata c u := by
        unfold KernelFootprint.Metadata
        rw [memory]
        exact NativeBindings.tile_metadata c o idx s metadata separate
      apply ih u metadataState
      · intro port off member
        rw [snap] at member
        exact reads port off (List.mem_append_right _ member)
      · intro job member i on
        exact writes job (List.mem_cons_of_mem idx member) i on

theorem io_safe (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (a : NativeArrays.Inputs c α) (metadata : NativeArrays.metadata c a)
    (bounds : RegionName → Nat) (backed : (programIO c o jobs).InBounds bounds a)
    (s : State α) (bound : (programIO c o jobs).InputBinding a s) :
    ScheduleSafe bounds ((programIO c o jobs).schedule a) s := by
  have snap:=binding_snapshot c o jobs a s bound
  have metadataState : KernelFootprint.Metadata c s := by
    rw [←NativeBindings.metadata_iff,snap]
    exact metadata
  apply schedule_safe c o jobs s metadataState separate bounds
  · intro port off member
    rw [snap] at member
    exact backed.2.1 port off member
  · intro idx member i on
    obtain ⟨j,hj,rfl⟩:=List.mem_iff_getElem.mp member
    exact backed.2.2.1 (0 : Fin 1) (⟨j,hj⟩,i) on

#axiomsClean tile_bounds
#axiomsClean schedule_safe
#axiomsClean io_safe
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
