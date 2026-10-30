import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Admission
import bench.tritonbench_g.context_attn_fwd.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealProof
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- Independent initial-array target, with finite future and padded denominator
weights and the actual Python host scale. No native floating exactness claim. -/
abbrev expected := Formula.expected

/-- Complete real source launch, derived from checked descriptor geometry.
The disjoint flat allocation represents logical tensor roles; physical
read-only aliasing is handled by the shared-byte input observation interface. -/
theorem complete (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (checked : Geometry.Checked c A.extent) (metadata : Geometry.Metadata c s)
    (separate : FPSuccess.Separate c) (allocation : A.Disjoint)
    (covered : ∀ rg,rg∉A.regions → A.extent rg=0) :
    Spec.Real (∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.flatSource c.args c.regions A Bool.false)
        c.grid (A.flattenState s) final) ∧
      (∀ idx i,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.active c.args c.regions s idx i →
        final.readMem
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).1
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).2 =
        VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.result c.args c.regions s idx i) ∧
      BlockState.WriteWithin
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.writes c.args c.regions A s)
        (A.flattenState s) final ∧
      (∀ rg off,c.regions.out≠rg → off<A.extent rg →
        final.mem A.flat (A.addr rg off)=(A.flattenState s).mem A.flat (A.addr rg off)) ∧
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.returned final).2=none) :=
  by
    have admitted := Geometry.admissible c A s checked metadata separate
    obtain ⟨final,hl,ho,hf,hr⟩ := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.complete
      c.args c.regions A s admitted allocation covered Bool.false
    refine ⟨final,hl,ho,hf,?_,hr⟩
    intro rg off hne bound
    exact (hf A.flat (A.addr rg off)
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.input_unwritten
        c.args c.regions A s admitted allocation covered rg hne off bound)).symm

theorem coverage (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (checked : Geometry.Checked c A.extent) (metadata : Geometry.Metadata c s)
    (separate : FPSuccess.Separate c) (b h row d : Nat)
    (hb : b<c.batches) (hh : h<c.layout.heads) (hr : row<c.length b) (hd : d<c.layout.D) :
    ∃ idx : GridIndex c.grid,∃ i : TileIndex [c.layout.BM,c.layout.D],
      c.active idx i ∧ Formula.batch c.layout (c.launchAt idx)=b ∧
      Formula.head c.layout (c.launchAt idx)=h ∧
      Formula.queryRow c.layout (c.launchAt idx) i.1.val=row ∧ i.2.1.val=d := by
  have ha := Geometry.admissible c A s checked metadata separate
  have hs : (row : Int)<(Program.inputs s c.regions).sequence b-(Program.inputs s c.regions).prompt b := by
    rw [metadata.2.1 b hb,metadata.2.2 b hb]
    simp only [Nat.cast_add,add_sub_cancel_right]
    exact_mod_cast hr
  obtain ⟨idx,i,hi,hb',hh',hr',hd'⟩ := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.covers
    c.args c.regions A s ha b h row d hb hh hs hd
  refine ⟨idx,i,?_,hb',hh',hr',hd'⟩
  change (Formula.queryRow c.layout (c.launchAt idx) i.1.val : Int)<
    (Program.inputs s c.regions).sequence (Formula.batch c.layout (c.launchAt idx))-
      (Program.inputs s c.regions).prompt (Formula.batch c.layout (c.launchAt idx)) at hi
  rw [metadata.2.1 _ (Host.batch_bound c idx),metadata.2.2 _ (Host.batch_bound c idx)] at hi
  change Formula.queryRow c.layout (c.launchAt idx) i.1.val<c.length _
  omega

theorem unique_writer (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (checked : Geometry.Checked c A.extent) (metadata : Geometry.Metadata c s)
    (separate : FPSuccess.Separate c) (idx jdx : GridIndex c.grid)
    (i j : TileIndex [c.layout.BM,c.layout.D]) (hi : c.active idx i) (hj : c.active jdx j)
    (he : c.offset idx i=c.offset jdx j) : idx=jdx ∧ i=j := by
  have ha := Geometry.admissible c A s checked metadata separate
  apply VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.unique_writer c.args c.regions A s ha idx jdx i j
  · change (Formula.queryRow c.layout (c.launchAt idx) i.1.val : Int)<Formula.sequence c.layout (c.launchAt idx) (Program.inputs s c.regions)
    rw [Formula.sequence,metadata.2.1 _ (Host.batch_bound c idx),metadata.2.2 _ (Host.batch_bound c idx)]
    simp only [Nat.cast_add,add_sub_cancel_right]
    exact_mod_cast hi
  · change (Formula.queryRow c.layout (c.launchAt jdx) j.1.val : Int)<Formula.sequence c.layout (c.launchAt jdx) (Program.inputs s c.regions)
    rw [Formula.sequence,metadata.2.1 _ (Host.batch_bound c jdx),metadata.2.2 _ (Host.batch_bound c jdx)]
    simp only [Nat.cast_add,add_sub_cancel_right]
    exact_mod_cast hj
  · change (((Program.inputs s c.regions).start (Formula.batch c.layout (c.launchAt idx))+
      Formula.queryRow c.layout (c.launchAt idx) i.1.val)*c.layout.outRow+
      Formula.head c.layout (c.launchAt idx)*c.layout.outHead+i.2.1.val*c.layout.outDim).toNat=_
    change _ = (((Program.inputs s c.regions).start (Formula.batch c.layout (c.launchAt jdx))+
      Formula.queryRow c.layout (c.launchAt jdx) j.1.val)*c.layout.outRow+
      Formula.head c.layout (c.launchAt jdx)*c.layout.outHead+j.2.1.val*c.layout.outDim).toNat
    rw [metadata.1 _ (Host.batch_bound c idx),metadata.1 _ (Host.batch_bound c jdx)]
    simpa [Host.Call.offset,Host.Call.output] using he

#print axioms complete
#print axioms unique_writer
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealProof
