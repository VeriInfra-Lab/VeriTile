import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.FiniteReadLocality
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOArray

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem table_preserves {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (left right : ι → Cell α) (a b : Memory α) (equal : ∀ i, left i=right i)
    (r : RegionName) (size : Nat) (same : Reads r size a b) :
    Reads r size (KernelReference.table region index left a) (KernelReference.table region index right b) := by
  intro off bound
  unfold KernelReference.table
  split_ifs with present
  · rw [equal]
  · exact same off bound

theorem table_covers {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (left right : ι → Cell α) (a b : Memory α) (equal : ∀ i, left i=right i)
    (r : RegionName) (size : Nat) (covers : ∀ off, off<size → ∃ i,region i=r ∧ index i=off) :
    Reads r size (KernelReference.table region index left a) (KernelReference.table region index right b) := by
  intro off bound
  unfold KernelReference.table
  rw [dif_pos (covers off bound),dif_pos (covers off bound),equal]

theorem state_value (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (a b : Memory α)
    (same : StateReads p rev seed a b) (i : KernelReference.StateCell p final) :
    KernelReference.stateValue M D p shape scale rev seed a i =
      KernelReference.stateValue M D p shape scale rev seed b i := by
  cases i with
  | history h t k v =>
      simp only [KernelReference.stateValue,state_values M D p (Ownership.statePID p h.val k.val v.val)
        a b scale rev seed h.isLt same]
  | endpoint enabled h k v =>
      simp only [KernelReference.stateValue,state_values M D p (Ownership.statePID p h.val k.val v.val)
        a b scale rev seed h.isLt same]

theorem history_coverage (p : Parameters) (rev final : Bool) (off : Nat)
    (bound : off<p.B*p.H*p.NT*p.K*p.V) :
    ∃ i : KernelReference.StateCell p final,
      KernelReference.stateRegion rev i=historyName rev ∧ KernelReference.stateIndex i=off := by
  let j := IOArray.index [p.B*p.H,p.NT,p.K,p.V]
    ⟨off,by simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using bound⟩
  refine ⟨.history j.1 j.2.1 j.2.2.1 j.2.2.2.1,rfl,?_⟩
  calc
    _ = TileShape.linearIndex [p.B*p.H,p.NT,p.K,p.V] j := by
      simp only [KernelReference.stateIndex,TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero]
      ring
    _ = off := IOArray.index_offset _ _

theorem endpoint_coverage (p : Parameters) (rev final : Bool) (enabled : final=Bool.true)
    (off : Nat) (bound : off<p.B*p.H*p.K*p.V) :
    ∃ i : KernelReference.StateCell p final,
      KernelReference.stateRegion rev i=finalName rev ∧ KernelReference.stateIndex i=off := by
  let j := IOArray.index [p.B*p.H,p.K,p.V]
    ⟨off,by simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using bound⟩
  refine ⟨.endpoint enabled j.1 j.2.1 j.2.2.1,rfl,?_⟩
  exact (IOArray.offset3 j.1 j.2.1 j.2.2.1).symm.trans (IOArray.index_offset _ _)

theorem state_preserves (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (a b : Memory α)
    (same : StateReads p rev seed a b) (r : RegionName) (size : Nat) (old : Reads r size a b) :
    Reads r size (KernelReference.state M D p shape scale rev seed final a)
      (KernelReference.state M D p shape scale rev seed final b) :=
  table_preserves _ _ _ _ a b (state_value M D p shape scale rev seed final a b same) r size old

theorem state_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (a b : Memory α)
    (same : StateReads p rev seed a b) :
    Reads (historyName rev) (p.B*p.H*p.NT*p.K*p.V)
      (KernelReference.state M D p shape scale rev seed final a)
      (KernelReference.state M D p shape scale rev seed final b) :=
  table_covers _ _ _ _ a b (state_value M D p shape scale rev seed final a b same)
    _ _ (history_coverage p rev final)

theorem state_endpoint (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (a b : Memory α)
    (same : StateReads p rev seed a b) (enabled : final=Bool.true) :
    Reads (finalName rev) (p.B*p.H*p.K*p.V)
      (KernelReference.state M D p shape scale rev seed final a)
      (KernelReference.state M D p shape scale rev seed final b) :=
  table_covers _ _ _ _ a b (state_value M D p shape scale rev seed final a b same)
    _ _ (endpoint_coverage p rev final enabled)

theorem output_value (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α)
    (same : OutputReads p a b) (i : KernelReference.OutputCell p) :
    KernelReference.outputValue M D p shape scale a i = KernelReference.outputValue M D p shape scale b i := by
  have valid := Ownership.output_valid p shape i.1.val i.2.1.val i.2.2.val i.1.isLt i.2.1.isLt i.2.2.isLt
  unfold KernelReference.outputValue
  dsimp only
  rw [output_result M D p _ a b scale valid.2.2 valid.2.1 same]

theorem output_coverage (p : Parameters) (off : Nat) (bound : off<p.B*p.H*p.T*p.V) :
    ∃ i : KernelReference.OutputCell p,("output" : RegionName)="output" ∧ KernelReference.outputIndex i=off := by
  let j := IOArray.index [p.B*p.H,p.T,p.V]
    ⟨off,by simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using bound⟩
  refine ⟨(j.1,j.2.1,j.2.2.1),rfl,?_⟩
  exact (IOArray.offset3 j.1 j.2.1 j.2.2.1).symm.trans (IOArray.index_offset _ _)

theorem output_preserves (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α) (same : OutputReads p a b)
    (r : RegionName) (size : Nat) (old : Reads r size a b) :
    Reads r size (KernelReference.output M D p shape scale a) (KernelReference.output M D p shape scale b) :=
  table_preserves _ _ _ _ a b (output_value M D p shape scale a b same) r size old

theorem output_output (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α) (same : OutputReads p a b) :
    Reads "output" (p.B*p.H*p.T*p.V)
      (KernelReference.output M D p shape scale a) (KernelReference.output M D p shape scale b) :=
  table_covers _ _ _ _ a b (output_value M D p shape scale a b same) _ _ (output_coverage p)

theorem gradient_value_table (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α)
    (same : GradientReads p a b) (i : KernelReference.GradientCell p) :
    KernelReference.gradientValue M D p shape scale a i = KernelReference.gradientValue M D p shape scale b i := by
  cases i with
  | query h t k =>
      simp only [KernelReference.gradientValue,
        (gradient_results M D p (Ownership.gradientPID p h.val t.val k.val) a b scale h.isLt same).1]
  | key h t k =>
      simp only [KernelReference.gradientValue,
        (gradient_results M D p (Ownership.gradientPID p h.val t.val k.val) a b scale h.isLt same).2]
  | valuePartial block h t v =>
      simp only [KernelReference.gradientValue,
        gradient_value M D p (Ownership.partialPID p block.val h.val t.val) a b scale h.isLt same]

theorem gradient_preserves (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α) (same : GradientReads p a b)
    (r : RegionName) (size : Nat) (old : Reads r size a b) :
    Reads r size (KernelReference.gradient M D p shape scale a) (KernelReference.gradient M D p shape scale b) :=
  table_preserves _ _ _ _ a b (gradient_value_table M D p shape scale a b same) r size old

theorem gradient_query_coverage (p : Parameters) (key : Bool) (off : Nat) (bound : off<p.B*p.H*p.T*p.K) :
    ∃ i : KernelReference.GradientCell p,
      KernelReference.gradientRegion i=(if key then ("keyGradient" : RegionName) else "queryGradient") ∧
      KernelReference.gradientIndex i=off := by
  let j := IOArray.index [p.B*p.H,p.T,p.K]
    ⟨off,by simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using bound⟩
  cases key
  · refine ⟨.query j.1 j.2.1 j.2.2.1,rfl,?_⟩
    exact (IOArray.offset3 j.1 j.2.1 j.2.2.1).symm.trans (IOArray.index_offset _ _)
  · refine ⟨.key j.1 j.2.1 j.2.2.1,rfl,?_⟩
    exact (IOArray.offset3 j.1 j.2.1 j.2.2.1).symm.trans (IOArray.index_offset _ _)

theorem gradient_partial_coverage (p : Parameters) (off : Nat)
    (bound : off<keyBlocks p*(p.B*p.H)*p.T*p.V) :
    ∃ i : KernelReference.GradientCell p,
      KernelReference.gradientRegion i="valuePartials" ∧ KernelReference.gradientIndex i=off := by
  let j := IOArray.index [keyBlocks p,p.B*p.H,p.T,p.V]
    ⟨off,by simpa only [TileShape.product,Nat.mul_one,Nat.mul_assoc] using bound⟩
  refine ⟨.valuePartial j.1 j.2.1 j.2.2.1 j.2.2.2.1,rfl,?_⟩
  calc
    _ = TileShape.linearIndex [keyBlocks p,p.B*p.H,p.T,p.V] j := by
      simp only [KernelReference.gradientIndex,TileShape.linearIndex,TileShape.product,Nat.mul_one,Nat.add_zero]
      ring
    _ = off := IOArray.index_offset _ _

theorem gradient_query (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α) (same : GradientReads p a b) (key : Bool) :
    Reads (if key then "keyGradient" else "queryGradient") (p.B*p.H*p.T*p.K)
      (KernelReference.gradient M D p shape scale a) (KernelReference.gradient M D p shape scale b) :=
  table_covers _ _ _ _ a b (gradient_value_table M D p shape scale a b same) _ _ (gradient_query_coverage p key)

theorem gradient_partial (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (a b : Memory α) (same : GradientReads p a b) :
    Reads "valuePartials" (keyBlocks p*(p.B*p.H)*p.T*p.V)
      (KernelReference.gradient M D p shape scale a) (KernelReference.gradient M D p shape scale b) :=
  table_covers _ _ _ _ a b (gradient_value_table M D p shape scale a b same) _ _ (gradient_partial_coverage p)

theorem sum_value (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (a b : Memory α) (same : Reads "valuePartials" (keyBlocks p*(p.B*p.H)*p.T*p.V) a b)
    (i : Fin (valueCount p)) :
    cast M (LibrarySum.evaluate M (selection.plans i)
      (fun k => (a "valuePartials" (k.val*valueCount p+i.val)).read .real)) =
    cast M (LibrarySum.evaluate M (selection.plans i)
      (fun k => (b "valuePartials" (k.val*valueCount p+i.val)).read .real)) := by
  congr 2
  funext k
  apply same
  have bound := packed_lt k.isLt i.isLt
  simpa only [valueCount,Nat.mul_assoc] using bound

theorem sum_preserves (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (a b : Memory α) (same : Reads "valuePartials" (keyBlocks p*(p.B*p.H)*p.T*p.V) a b)
    (r : RegionName) (size : Nat) (old : Reads r size a b) :
    Reads r size (KernelReference.sum M p selection a) (KernelReference.sum M p selection b) :=
  table_preserves _ _ _ _ a b (fun i => congrArg (Cell.mk .real) (sum_value M p selection a b same i)) r size old

theorem sum_output (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (a b : Memory α) (same : Reads "valuePartials" (keyBlocks p*(p.B*p.H)*p.T*p.V) a b) :
    Reads "valueGradient" (valueCount p) (KernelReference.sum M p selection a) (KernelReference.sum M p selection b) :=
  table_covers _ _ _ _ a b (fun i => congrArg (Cell.mk .real) (sum_value M p selection a b same i))
    _ _ (fun off bound => ⟨⟨off,bound⟩,rfl,rfl⟩)

#axiomsClean state_value
#axiomsClean state_history
#axiomsClean state_endpoint
#axiomsClean output_value
#axiomsClean output_output
#axiomsClean gradient_value_table
#axiomsClean gradient_query
#axiomsClean gradient_partial
#axiomsClean sum_output
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality
