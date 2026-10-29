import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Execution

namespace VeriTile.Bench.Repairs.UniformSampling.Schedule
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
open VeriTile.Bench.TritonBenchG.UniformSampling
open IndexProof
noncomputable section
open scoped Classical

theorem unique_output {n start : Nat} (hs:block n∣start) {x y : Location (block n)}
    (hx:x∈schedule n start) (hy:y∈schedule n start) (he:globalIndex x=globalIndex y) : x=y := by
  have xl:=launch_bounds hs hx
  have yl:=launch_bounds hs hy
  have hc:x.1=y.1:=ChunkProof.unique start (counterCount n) 0 (globalIndex x/4)
    (schedule_mem.mp hx).1 (schedule_mem.mp hy).1
    (by unfold globalIndex;omega) (by unfold globalIndex;omega)
    (by rw [he];unfold globalIndex;omega) (by rw [he];unfold globalIndex;omega)
  have he':outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2=
      outputIndex y.2.1 (block n) y.2.2.1 y.2.2.2:=by unfold globalIndex at he;rw [hc] at he;omega
  have hu:=SourceDomain.output_unique (ReservationProof.block_properties n).2.1
    x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2 he'
  exact Prod.ext hc (Prod.ext hu.1 (Prod.ext hu.2.1 hu.2.2))

def copyEvents (M : Arithmetic α) (v : View n) (values : Nat→α) (order : List Nat) : List (Write α) :=
  order.map fun i=>⟨v.address i,physical M v (values i)⟩

def copy (api : API α Store Byte) (M : Arithmetic α) (v : View n) (values : Nat→α)
    (order : List Nat) (m : Nat→Store) : Nat→Store :=
  Function.update m v.region (Memory.run api (copyEvents M v values order) (m v.region))

theorem copy_readback (api : API α Store Byte) (M : Arithmetic α) (v : View n) (values : Nat→α)
    (order : List Nat) (same:∀i,i∈order↔i<n) (m : Nat→Store) {i : Nat} (hi:i<n) :
    api.read (copy api M v values order m v.region) (v.address i)=physical M v (values i) := by
  simp only [copy,Function.update_self]
  apply run_readback
  · refine ⟨⟨v.address i,physical M v (values i)⟩,?_,rfl,rfl⟩
    exact List.mem_map.mpr ⟨i,(same i).mpr hi,rfl⟩
  · intro e he
    obtain ⟨j,hj,rfl⟩:=List.mem_map.mp he
    by_cases h:j=i
    · subst j;exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (v.separate j i ((same j).mp hj) hi h)

theorem copy_prefix_frame (api : API α Store Byte) (M : Arithmetic α) (v : View n) (values : Nat→α)
    (order : List Nat) (inside:∀i,i∈order→i<n) (m : Nat→Store) (r b : Nat)
    (outside:Execution.Outside v r b) :
    api.byte (copy api M v values order m r) b=api.byte (m r) b := by
  by_cases hr:r=v.region
  · subst r
    simp only [copy,Function.update_self]
    apply run_byte_frame
    intro e he
    obtain ⟨i,hi,rfl⟩:=List.mem_map.mp he
    exact (outside.resolve_left (not_not_intro rfl)) i (inside i hi)
  · simp [copy,Function.update_of_ne hr]

def run (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars) (seed start : Nat)
    (scratch : Scratch n) (v : View n) (nativeOrder : List (Location (block n)))
    (copyOrder : List Nat) (m : Nat→Store) : Nat→Store :=
  let native:=Native.run api M scalars seed start scratch nativeOrder (m scratch.region)
  let afterNative:=Function.update m scratch.region native
  copy api M v (fun i=>api.read native (scratchAddress scratch i)) copyOrder afterNative

/-- Successful observations are independent of legal native and copy order.
The failed variants below retain arbitrary completed prefixes of those orders. -/
theorem readback (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars) (seed start : Nat)
    (scratch : Scratch n) (v : View n) (nativeOrder : List (Location (block n)))
    (copyOrder : List Nat) (m : Nat→Store) (hs:block n∣start)
    (nativeSame:∀x,x∈nativeOrder↔x∈schedule n start) (copySame:∀i,i∈copyOrder↔i<n)
    (neg:∀x,M.neg32 (M.neg32 x)=x) {i : Nat} (hi:i<n) :
    logical api M (run api M scalars seed start scratch v nativeOrder copyOrder m) v i=
      Native.result M scalars seed start (block n) i := by
  unfold logical run
  rw [copy_readback api M v _ copyOrder copySame _ hi,
    Native.serialized_readback api M scalars seed start scratch nativeOrder nativeSame _ hs hi]
  cases h:v.negative <;> simp [physical,h,neg]

theorem arbitrary_prefix_frame (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (v : View n)
    (nativePrefix : List (Location (block n))) (copyPrefix : List Nat)
    (copyInside:∀i,i∈copyPrefix→i<n) (m : Nat→Store) (r b : Nat)
    (fresh:r≠scratch.region) (outside:Execution.Outside v r b) :
    api.byte (run api M scalars seed start scratch v nativePrefix copyPrefix m r) b=api.byte (m r) b := by
  unfold run
  rw [copy_prefix_frame api M v _ copyPrefix copyInside _ r b outside]
  simp [Function.update_of_ne fresh]

#axiomsClean unique_output
#axiomsClean readback
#axiomsClean arbitrary_prefix_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Schedule
