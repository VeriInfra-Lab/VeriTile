import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Contract
import bench.repairs.tritonbench_g.token_attn_reduceV.Stores

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
open VeriTile.Triton FP.Structural TokenAttnReduceV
noncomputable section
set_option maxHeartbeats 3000000

def parameters (m : Inputs) : Parameters where
  prob := "probability"
  value := "value"
  output := "output"
  tokens := Region.cast ("pages" : RegionName)
  request := Region.cast ("request" : RegionName)
  start := Region.cast ("start" : RegionName)
  length := Region.cast ("length" : RegionName)
  requestStride := 1
  startStride := 1
  lengthStride := 1
  requestBatchStride := columns m
  requestTokenStride := 1
  probHeadStride := tokens m
  probTokenStride := 1
  valueBatchStride := kvheads m*width m
  valueHeadStride := width m
  valueDimStride := 1
  outputBatchStride := heads m*width m
  outputHeadStride := width m
  outputDimStride := 1
  kvGroup := group m
  width := width m
  block := 128

abbrev Cells (α : Type) := Nat → Cell α

/-- Direct typed loads from the original caller pointers. A single metadata
reader retains all legal read aliases; no host snapshot or packed copy exists. -/
def view {α : Type} [Inhabited α] (m : Inputs) (read : Nat → α)
    (metadata : Reader) (before : Cells α) : Grid.Memory α := fun r i =>
  if r="probability" then .mk .fp32 (read (base m .probability+4*i))
  else if r="value" then .mk .fp32 (read (base m .value+4*i))
  else if r="pages" then .mk .nat (index m metadata .pages i)
  else if r="request" then .mk .nat (index m metadata .request i)
  else if r="start" then .mk .nat (index m metadata .start i)
  else if r="length" then .mk .nat (index m metadata .length i)
  else if r="output" then before (base m .output+4*i)
  else .mk .nat 0

def outputIndex (m : Inputs) (b h d : Nat) := (b*heads m+h)*width m+d

theorem offset_dense (m : Inputs) (pid : Grid.PID) (i : TileIndex [(parameters m).width]) :
    Grid.offset (parameters m) pid i=outputIndex m pid.1 pid.2 i.1.val := by
  simp [Grid.offset,parameters,outputIndex,Nat.add_mul,Nat.mul_assoc]

theorem pair_injective (n a b c d : Nat) (hb : b<n) (hd : d<n)
    (same : a*n+b=c*n+d) : a=c ∧ b=d := by
  have positive : 0<n := by omega
  have rem := congrArg (fun k => k%n) same
  have equal : b=d := by simpa [Nat.add_mod,Nat.mod_eq_of_lt hb,Nat.mod_eq_of_lt hd] using rem
  refine ⟨?_,equal⟩
  apply Nat.eq_of_mul_eq_mul_right positive
  omega

theorem unique (m : Inputs) (work : List Grid.PID)
    (order : (Grid.fullGrid (batches m) (heads m)).Perm work) :
    Grid.Unique (parameters m) work := by
  intro e he f hf same
  have eh := (Grid.mem_fullGrid _ _ _).mp (order.mem_iff.mpr ((Grid.mem_events _ _ _).mp he))
  have fh := (Grid.mem_fullGrid _ _ _).mp (order.mem_iff.mpr ((Grid.mem_events _ _ _).mp hf))
  rw [offset_dense,offset_dense] at same
  have inner := pair_injective (width m) _ _ _ _ e.2.1.isLt f.2.1.isLt same
  have outer := pair_injective (heads m) _ _ _ _ eh.2 fh.2 inner.1
  exact Prod.ext (Prod.ext outer.1 outer.2)
    (Prod.ext (Fin.ext inner.2) (Subsingleton.elim _ _))

theorem separate (m : Inputs) : Grid.Separate (parameters m) := by
  intro r hr
  simp only [Grid.Reads,parameters,Region.cast] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;> change _≠("output" : RegionName) <;> decide

def RawFrame {α : Type} (m : Inputs) (before after : Grid.Memory α) : Prop :=
  ∀ (r : RegionName) (i : Nat),(r≠"output" ∨ ∀ b<batches m,∀ h<heads m,∀ d<width m,
    i≠outputIndex m b h d) → after r i=before r i

/-- Whole success and every output, including zero-length rows. The original
single source quotation, dynamic loop and stores are executed unchanged. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (m : Inputs) (memory : Grid.Memory α) (work : List Grid.PID)
    (order : (Grid.fullGrid (batches m) (heads m)).Perm work) :
    ∃ after,Grid.run M R (parameters m) work memory=some after ∧
      (∀ b<batches m,∀ h<heads m,∀ d : Fin (width m),
        after "output" (outputIndex m b h d.val)=
          .mk .fp32 (outputValues M R (parameters m) (Grid.state memory (b,h)) (d,PUnit.unit))) ∧
      RawFrame m memory after := by
  obtain ⟨after,run,_,values,frame⟩ := Grid.whole M R (parameters m) work memory
    (separate m) (unique m work order)
  refine ⟨after,run,?_,?_⟩
  · intro b hb h hh d
    let e : Grid.Event (parameters m) := ((b,h),(d,PUnit.unit))
    have live : e∈Grid.events (parameters m) work :=
      (Grid.mem_events _ _ _).mpr (order.mem_iff.mp ((Grid.mem_fullGrid _ _ _).mpr ⟨hb,hh⟩))
    simpa [offset_dense,Grid.values,e] using values e live
  · intro r i outside
    apply frame
    rcases outside with different|miss
    · exact Or.inl different
    · refine Or.inr ?_
      intro e he
      have legal := (Grid.mem_fullGrid _ _ _).mp (order.mem_iff.mpr ((Grid.mem_events _ _ _).mp he))
      rw [offset_dense]
      exact miss e.1.1 legal.1 e.1.2 legal.2 e.2.1.val e.2.1.isLt

theorem linear_bound (a n b k : Nat) (ha : a<n) (hb : b<k) : a*k+b<n*k := by
  calc
    a*k+b < a*k+k := Nat.add_lt_add_left hb _
    _ = (a+1)*k := by simp [Nat.add_mul]
    _ ≤ n*k := Nat.mul_le_mul_right k ha

theorem output_bound (m : Inputs) (metadata : Reader) (contract : InputContract m metadata)
    (b h d : Nat) (hb : b<batches m) (hh : h<heads m) (hd : d<width m) :
    outputIndex m b h d<(m .output).numel := by
  have bound := linear_bound _ _ _ _ (linear_bound _ _ _ _ hb hh) hd
  simpa [outputIndex,VeriTile.Torch.TensorMetadata.numel,contract.header.outputShape,Nat.mul_assoc] using bound

/-- Physical readback of source stores only. Every exterior word remains the
original caller word; this is not an allocation or a copyback operation. -/
def observe {α β : Type} [Inhabited α] (read : α → β) (m : Inputs)
    (before : Nat → β) (after : Grid.Memory α) (address : Nat) : β :=
  if base m .output≤address ∧ address<base m .output+4*(m .output).numel ∧
      (address-base m .output)%4=0
  then read ((after "output" ((address-base m .output)/4)).read .fp32)
  else before address

theorem observe_store {α β : Type} [Inhabited α] (read : α → β) (m : Inputs)
    (before : Nat → β) (after : Grid.Memory α) (i : Nat) (hi : i<(m .output).numel) :
    observe read m before after (base m .output+4*i)=read ((after "output" i).read .fp32) := by
  simp [observe,show base m .output+4*i<base m .output+4*(m .output).numel by omega]

def Exterior (m : Inputs) (address : Nat) :=
  address<base m .output ∨ base m .output+4*(m .output).numel≤address

theorem observe_frame {α β : Type} [Inhabited α] (read : α → β) (m : Inputs)
    (before : Nat → β) (after : Grid.Memory α) (address : Nat) (outside : Exterior m address) :
    observe read m before after address=before address := by
  unfold observe
  rw [if_neg (by unfold Exterior at outside;omega)]

#axiomsClean complete
#axiomsClean output_bound
#axiomsClean observe_store
#axiomsClean observe_frame
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
