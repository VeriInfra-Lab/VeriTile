import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.ProgramResult

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem region_output (s : Shape) (r : RotaryTransform.Region) :
    KernelAlgebra.regionName r=KernelAlgebra.regionName s.output ↔ r=s.output := by
  cases hi : s.inplace <;> cases r <;> simp [KernelAlgebra.regionName,Shape.output,hi]

theorem program_active (c : Config) (u : State α) (w : Work)
    (hw : w∈programWork c (pid u)) : c.Active w :=
  of_decide_eq_true (List.mem_filter.mp hw).2

theorem work_member (c : Config) (u : State α) (lane pair : Nat)
    (hl : lane<c.shape.blockM) (hp : pair<c.shape.blockK/2)
    (live : c.Active (work c.shape u lane pair)) :
    work c.shape u lane pair∈programWork c (pid u) := by
  apply List.mem_filter.mpr
  refine ⟨List.mem_map.mpr ⟨(lane,pair),?_,rfl⟩,by simpa using live⟩
  simp [lanes,hl,hp]

theorem work_member_live (c : Config) (valid : c.Valid) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (lane pair : Nat) (hl : lane<c.shape.blockM)
    (live : active c.shape u lane pair Bool.false) :
    work c.shape u lane pair∈programWork c (pid u) :=
  work_member c u lane pair hl (lt_of_lt_of_le live.2 (blockK_covers c valid))
    ((live_eq c valid u md batch head lane pair).mp live)

theorem result_work (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (w : Work) (hw : w∈programWork c (pid u)) (odd : Bool) :
    (result c.shape p u).mem (KernelAlgebra.regionName c.shape.output) (c.cell w odd)=
      Cell.mk .real (component (pairValues c p w (KernelAlgebra.memory u)) odd) := by
  obtain ⟨mapped,live⟩ := List.mem_filter.mp hw
  obtain ⟨⟨lane,pair⟩,hl,rfl⟩ := List.mem_map.mp mapped
  have activeW : c.Active (work c.shape u lane pair) := of_decide_eq_true live
  have laneBound : lane<c.shape.blockM := (List.mem_product.mp hl).1 |> List.mem_range.mp
  exact result_pair c valid p u md batch head lane pair laneBound
    ((live_eq c valid u md batch head lane pair).mpr activeW) odd

theorem result_frame (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (r : RotaryTransform.Region) (o : Nat)
    (outside : ∀ w∈programWork c (pid u),¬Writes c w r o) :
    (result c.shape p u).mem (KernelAlgebra.regionName r) o=u.mem (KernelAlgebra.regionName r) o := by
  by_cases region : r=c.shape.output
  swap
  · exact result_other c.shape p u _ o (fun eq => region ((region_output _ _).mp eq))
  subst r
  unfold result
  split
  · rfl
  · split
    · rename_i inter
      apply scatter_frame
      right
      intro i live eq
      have bound : i.2.1.val<2*c.shape.half := live.2
      obtain ⟨hp,back⟩ := column_inverse c.shape i.2.1.val bound
      let pair := c.shape.pairOf i.2.1.val
      let w := work c.shape u i.1.val pair
      have live' : active c.shape u i.1.val pair Bool.false := ⟨live.1,hp⟩
      have hw := work_member_live c valid u md batch head _ _ i.1.isLt live'
      apply outside w hw
      refine ⟨rfl,?_⟩
      have cell : o=c.cell w (c.shape.second i.2.1.val) := by
        rw [←offset_dim c u md batch]
        simpa only [pair,back,Nat.add_zero] using eq
      cases sec : c.shape.second i.2.1.val
      · exact Or.inl (by simpa only [sec] using cell)
      · exact Or.inr (by simpa only [sec] using cell)
    · rename_i split
      have split' : c.shape.interleaved=Bool.false := Bool.eq_false_iff.mpr split
      have miss (second : Bool) :
          ∀ i : TileIndex [c.shape.blockM,c.shape.blockK/2],
            active c.shape u i.1.val i.2.1.val Bool.false →
              o≠dataOffset c.shape u i.1.val i.2.1.val+(if second then c.shape.half else 0) := by
        intro i live eq
        have hw := work_member_live c valid u md batch head _ _ i.1.isLt live
        apply outside _ hw
        refine ⟨rfl,?_⟩
        have cell : o=c.cell (work c.shape u i.1.val i.2.1.val) second := by
          rw [←offset_dim c u md batch]
          simpa [Shape.dim,split',dataOffset,Nat.add_assoc] using eq
        cases second
        · exact Or.inl cell
        · exact Or.inr cell
      dsimp only
      rw [scatter_frame c.shape u _ _ Bool.false c.shape.half _ _ _
        (Or.inr (by simpa using miss Bool.true)),
        scatter_frame c.shape u _ _ Bool.false 0 _ _ _ (Or.inr (by simpa using miss Bool.false))]

/-- The actual matrix source program and the existing serial-pair model have
the same memory effect. No successful-output fact appears among the premises. -/
theorem source_program (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) :
    KernelAlgebra.memory (result c.shape p u)=sourceProgram c p (pid u) (KernelAlgebra.memory u) := by
  classical
  funext r o
  by_cases touched : ∃ w∈programWork c (pid u),Writes c w r o
  · obtain ⟨w,hw,region,cell|cell⟩ := touched
    all_goals subst r o
    · rw [show sourceProgram c p (pid u) (KernelAlgebra.memory u) c.shape.output (c.cell w Bool.false)=
          component (pairValues c p w (KernelAlgebra.memory u)) Bool.false from
        vector_readback c valid p _ (program_active c u) _ w hw Bool.false]
      simp only [KernelAlgebra.memory,result_work c valid p u md batch head w hw Bool.false,Cell.read_same]
    · rw [show sourceProgram c p (pid u) (KernelAlgebra.memory u) c.shape.output (c.cell w Bool.true)=
          component (pairValues c p w (KernelAlgebra.memory u)) Bool.true from
        vector_readback c valid p _ (program_active c u) _ w hw Bool.true]
      simp only [KernelAlgebra.memory,result_work c valid p u md batch head w hw Bool.true,Cell.read_same]
  · have outside : ∀ w∈programWork c (pid u),¬Writes c w r o := by simpa using touched
    rw [show sourceProgram c p (pid u) (KernelAlgebra.memory u) r o=KernelAlgebra.memory u r o from
      vector_frame c p _ _ r o outside]
    exact congrArg (Cell.read .real) (result_frame c valid p u md batch head r o outside)

#axiomsClean result_work
#axiomsClean result_frame
#axiomsClean source_program
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
