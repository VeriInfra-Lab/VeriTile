import bench.tritonbench_g.rope_embedding.Proofs.Kernel
namespace VeriTile.Bench.Repairs.RopeEmbedding.KernelFP
open VeriTile.Triton FP.Structural
open Numerics
noncomputable section
set_option maxHeartbeats 3000000

/-- Raw memory is interpreted at the explicit compiler-lowering boundary.
Product tags are ephemeral register information; memory starts with bare words. -/
def initial (memory : RegionName→Nat→Word) : State Tagged where
  regs := fun _ _ _ => none
  mem := fun region o => .mk .real (leaf (memory region o))
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def words (s : State Tagged) (region : RegionName) (o : Nat) : Word :=
  ((s.mem region o).read .real).word

/-- Independent native output, with precisely the original coefficient-first
mul/FMA operands. The chosen coordinate is unique by packed addressing. -/
def nativeMemory (p : Primitives) (r : Request) (back : Bool)
    (row group : Nat) (memory : RegionName→Nat→Word) (region : RegionName) (o : Nat) : Word := by
  classical
  exact if hw : Kernel.Written r row group region o then
    let h := Classical.choose hw.2
    let c := Classical.choose (Classical.choose_spec hw.2)
    Numerics.result p r back (memory "Q") (memory "C") (memory "S") row h c
  else memory region o

/-- This equation is the successful implementation relation, over every
FP32 memory word. The reference is an independent native instruction graph,
not the source executor or an identity program. -/
theorem successful_equivalence (p : Primitives) (r : Request) (back : Bool)
    (memory : RegionName→Nat→Word) (row group : Nat) (hr : row<r.rows) (hg : group<r.groups) :
    ∃ final, FP.Structural.exec (operations p).backend (Source.source (parameters r) back)
        (programState (initial memory) row group)=some final ∧
      words final=nativeMemory p r back row group memory ∧
      Kernel.Post r row group (initial memory) final (fun h c =>
        scalarResult (operations p) r back (fun o => leaf (memory "Q" o))
          (fun o => leaf (memory "C" o)) (fun o => leaf (memory "S" o)) row h c) := by
  classical
  obtain ⟨final,he,hpost⟩ := Kernel.executes (operations p) r back (initial memory) row group hr hg
  refine ⟨final,he,?_,hpost⟩
  funext region o
  unfold nativeMemory
  split
  · rename_i hw
    let h := Classical.choose hw.2
    let c := Classical.choose (Classical.choose_spec hw.2)
    have info := Classical.choose_spec (Classical.choose_spec hw.2)
    change group*4≤h ∧ h<min (group*4+4) r.heads ∧ c<2*r.half ∧ o=r.offset row h c at info
    change words final region o=Numerics.result p r back (memory "Q") (memory "C") (memory "S") row h c
    unfold words
    rw [hw.1,info.2.2.2,hpost.values h c info.1 info.2.1 info.2.2.1]
    simpa only [initial,Cell.read,leaf] using source_profile p r back
      (fun o => leaf (memory "Q" o)) (fun o => leaf (memory "C" o))
      (fun o => leaf (memory "S" o)) row h c
  · rename_i outside
    rw [words,hpost.frame region o outside]
    rfl

/-- The public concrete instantiation fixes IEEE/FTZ/cast/FMA semantics in cfg;
there are no numerical rewrite assumptions or finite-input restrictions. -/
theorem fp32_successful_equivalence (cfg : FP.Config) (r : Request) (back : Bool)
    (memory : RegionName→Nat→Word) (row group : Nat) (hr : row<r.rows) (hg : group<r.groups) :
    ∃ final, FP.Structural.exec (operations (softwarePrimitives cfg)).backend
        (Source.source (parameters r) back) (programState (initial memory) row group)=some final ∧
      words final=nativeMemory (softwarePrimitives cfg) r back row group memory := by
  obtain ⟨final,he,hm,_⟩ := successful_equivalence (softwarePrimitives cfg) r back memory row group hr hg
  exact ⟨final,he,hm⟩

#specNonCircular nativeMemory avoiding [Source.source,FP.Structural.exec,Source.groupResult,Source.value]
#axiomsClean successful_equivalence
#axiomsClean fp32_successful_equivalence
end
end VeriTile.Bench.Repairs.RopeEmbedding.KernelFP
