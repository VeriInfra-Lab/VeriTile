import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.NativeIO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

theorem lookup_eq (addresses : List Nat) (values : Fin addresses.length → NativeGraph.Expr)
    (read : Nat → NativeGraph.Expr) (off : Nat) (member : off∈addresses)
    (bound : ∀ i,read addresses[i]=values i) : lookup addresses values off=read off := by
  obtain ⟨i,hi,ho⟩ := List.getElem_of_mem member
  have found : ∃ i : Fin addresses.length,addresses[i]=off := ⟨⟨i,hi⟩,ho⟩
  rw [lookup,dif_pos found]
  rw [←bound (Classical.choose found),Classical.choose_spec found]

theorem inner_binding_key (c : HostFP.Inner) (x : (innerIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (innerIO c).InputBinding x s) (off : Nat) (member : off∈collect (keyReads c)) :
    (s.mem "k" off).read .real=((innerInput c x).mem "k" off).read .real := by
  simp only [innerInput, show ("k" : RegionName) ≠ "v" by decide,
    show ("v" : RegionName) ≠ "k" by decide, show ("g" : RegionName) ≠ "k" by decide,
    show ("g" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "k" by decide,
    show ("h0" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "g" by decide,
    ↓reduceIte,Cell.read_same]
  exact (lookup_eq _ _ (fun o=>(s.mem "k" o).read .real) off member
    (fun i=>bound (0 : Fin 4) (i,PUnit.unit) trivial)).symm

theorem inner_binding_value (c : HostFP.Inner) (x : (innerIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (innerIO c).InputBinding x s) (off : Nat) (member : off∈collect (valueReads c)) :
    (s.mem "v" off).read .real=((innerInput c x).mem "v" off).read .real := by
  simp only [innerInput, show ("k" : RegionName) ≠ "v" by decide,
    show ("v" : RegionName) ≠ "k" by decide, show ("g" : RegionName) ≠ "k" by decide,
    show ("g" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "k" by decide,
    show ("h0" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "g" by decide,
    ↓reduceIte,Cell.read_same]
  exact (lookup_eq _ _ (fun o=>(s.mem "v" o).read .real) off member
    (fun i=>bound (1 : Fin 4) (i,PUnit.unit) trivial)).symm

theorem inner_binding_gate (c : HostFP.Inner) (x : (innerIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (innerIO c).InputBinding x s) (off : Nat) (member : off∈collect (gateReads c)) :
    (s.mem "g" off).read .real=((innerInput c x).mem "g" off).read .real := by
  simp only [innerInput, show ("k" : RegionName) ≠ "v" by decide,
    show ("v" : RegionName) ≠ "k" by decide, show ("g" : RegionName) ≠ "k" by decide,
    show ("g" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "k" by decide,
    show ("h0" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "g" by decide,
    ↓reduceIte,Cell.read_same]
  exact (lookup_eq _ _ (fun o=>(s.mem "g" o).read .real) off member
    (fun i=>bound (2 : Fin 4) (i,PUnit.unit) trivial)).symm

theorem inner_binding_seed (c : HostFP.Inner) (x : (innerIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (innerIO c).InputBinding x s) (off : Nat) (member : off∈collect (seedReads c)) :
    (s.mem "h0" off).read .fp32=((innerInput c x).mem "h0" off).read .fp32 := by
  simp only [innerInput, show ("k" : RegionName) ≠ "v" by decide,
    show ("v" : RegionName) ≠ "k" by decide, show ("g" : RegionName) ≠ "k" by decide,
    show ("g" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "k" by decide,
    show ("h0" : RegionName) ≠ "v" by decide, show ("h0" : RegionName) ≠ "g" by decide,
    ↓reduceIte,Cell.read_same]
  exact (lookup_eq _ _ (fun o=>(s.mem "h0" o).read .fp32) off member
    (fun i=>bound (3 : Fin 4) (i,PUnit.unit) trivial)).symm

theorem inner_binding (c : HostFP.Inner) (x : (innerIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (innerIO c).InputBinding x s) (j : Job c.plans) :
    innerStores c j s=innerStores c j (innerInput c x) := by
  apply NativeStores.inner_eq
  refine ⟨rfl,?_,?_,?_,?_,?_,?_,?_⟩
  · intro n hn i
    exact inner_binding_key c x s bound _ (mem_collect _ j (key_mem c j s n hn i))
  · intro n hn i
    exact inner_binding_value c x s bound _ (mem_collect _ j (value_mem c j s n hn i))
  · intro hg n hn i
    exact inner_binding_gate c x s bound _ (mem_collect _ j (gate_key_mem c j s hg n hn i))
  · intro hg n hn i
    exact inner_binding_gate c x s bound _ (mem_collect _ j (gate_value_mem c j s hg n hn i))
  · intro hg n hn i
    exact inner_binding_gate c x s bound _ (mem_collect _ j (key_end_mem c j s hg n hn i))
  · intro hg n hn i
    exact inner_binding_gate c x s bound _ (mem_collect _ j (value_end_mem c j s hg n hn i))
  · intro hs i
    exact inner_binding_seed c x s bound _ (mem_collect _ j (seed_mem c j s hs i))

theorem prefix_binding (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) (s : State NativeGraph.Expr)
    (bound : (prefixIO c).InputBinding x s) (j : Job c.plans) :
    prefixStores c j s=prefixStores c j (prefixInput c x) := by
  apply NativeStores.prefix_eq
  refine ⟨rfl,?_⟩
  intro active i
  have member := mem_collect _ j (prefix_mem c j s active i)
  simp only [atJob,HostFP.gridState,prefixInput,↓reduceIte,Cell.read_same]
  exact (lookup_eq _ _ (fun o=>(s.mem "s" o).read .fp32) _ member
    (fun i=>bound (0 : Fin 1) (i,PUnit.unit) trivial)).symm

#axiomsClean inner_binding
#axiomsClean prefix_binding
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
