import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Runtime

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TypedGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform Runtime
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

/-- An active source write retains the explicit FP32 cell tag. -/
theorem block_tag (ops : Operations α) (r : Request) (X Y : RegionName)
    (q : Program) (s : State α) (o : Nat) (hit : Hits r q o) :
    ∃ v, (block ops r X Y q s).mem Y o = Cell.mk .fp32 v := by
  obtain ⟨i,im,ha,he⟩ := hit
  have hv := Source.written_readback ops.backend (parameters r X Y) (setup s q)
    (source_apart r X Y (setup s q)) im i (source_active r X Y s (location q i im) ha)
  have off : Source.outputOffset (parameters r X Y) (setup s q) im i = o :=
    ((source_offset r X Y (setup s q) (location q i im)).2).trans he.symm
  rw [off] at hv
  exact ⟨_,hv⟩

/-- Later source programs either preserve the cell or write another FP32 cell. -/
theorem compose_keeps_tag (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) (o : Nat)
    (valid : ∀ q∈qs,q.1<r.batches)
    (typed : ∃ v,s.mem Y o=Cell.mk .fp32 v) :
    ∃ v,(compose ops r X Y qs s).mem Y o=Cell.mk .fp32 v := by
  induction qs generalizing s with
  | nil => exact typed
  | cons q rest ih =>
    apply ih
    · exact fun p hp => valid p (List.mem_cons_of_mem _ hp)
    · by_cases hit : Hits r q o
      · exact block_tag ops r X Y q s o hit
      · simpa only [block_frame ops r X Y q s Y o
          (valid q (List.mem_cons_self ..)) (Or.inr hit)] using typed

/-- Full-grid coverage proves every requested output's tag, without a tag premise. -/
theorem compose_tag (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) (o : Nat)
    (valid : ∀ q∈qs,q.1<r.batches) (covered : ∃ q∈qs,Hits r q o) :
    ∃ v,(compose ops r X Y qs s).mem Y o=Cell.mk .fp32 v := by
  induction qs generalizing s with
  | nil => simp at covered
  | cons q rest ih =>
    by_cases hit : Hits r q o
    · exact compose_keeps_tag ops r X Y rest _ o
        (fun p hp => valid p (List.mem_cons_of_mem _ hp)) (block_tag ops r X Y q s o hit)
    · apply ih
      · exact fun p hp => valid p (List.mem_cons_of_mem _ hp)
      · obtain ⟨p,hp,h⟩ := covered
        exact ⟨p,(List.mem_cons.mp hp).resolve_left (by intro he; subst p; exact hit h),h⟩

/-- The existing full execution/value/frame theorem, retaining exact output tags. -/
theorem whole (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (all : qs.Perm (programs r)) (s : State α) (different : X≠Y) :
    ∃ final,run ops r X Y qs s=some final ∧
      (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
        final.mem Y (r.offset b m c)=Cell.mk .fp32
          (scalarResult ops r (fun o => (s.mem X o).read .fp32) b m c)) ∧
      (∀ region o,region≠Y ∨ r.size≤o → final.mem region o=s.mem region o) := by
  obtain ⟨final,ran,values,frame⟩ := Runtime.whole ops r X Y qs all s different
  obtain ⟨actual,executed,hm⟩ := Runtime.complete ops r X Y qs s
  have same : actual=final := Option.some.inj (executed.symm.trans ran)
  subst actual
  refine ⟨final,ran,?_,frame⟩
  intro b m c hb hr hc
  have covered : ∃ q∈qs,Hits r q (r.offset b m c) := by
    obtain ⟨l,hl,ha,hb',hr',hc'⟩ := coverage r b m c hb hr hc
    refine ⟨(batch l,program l),all.mem_iff.mpr ((mem_programs r _).mpr ((mem_schedule r l).mp hl)),?_⟩
    refine ⟨lane l,imaginary l,ha,?_⟩
    change r.offset b m c=r.offset (batch l) (row r l) (column r l)
    rw [hb',hr',hc']
  obtain ⟨v,hv⟩ := compose_tag ops r X Y qs s (r.offset b m c)
    (fun q hq => ((mem_programs r q).mp (all.mem_iff.mp hq)).1) covered
  rw [← hm] at hv
  have value := values b m c hb hr hc
  rw [hv,Cell.read_same] at value
  exact hv.trans (congrArg (Cell.mk .fp32) value)

#axiomsClean whole
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.TypedGrid
