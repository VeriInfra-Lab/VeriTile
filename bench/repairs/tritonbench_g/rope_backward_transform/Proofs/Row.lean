import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Spec
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def state {α : Type} [Inhabited α] (m : Source.Memory α) (row : Nat) : State α where
  mem := m
  regs := fun _ _ _ => none
  pids := fun a => if a=0 then row else 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d
def rowMemory {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (row : Nat) (m : Source.Memory α) : Source.Memory α :=
  Source.referenceMemory M .fp32 .fp32 (parameters r) (state m row)

theorem row_execution {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (row : Nat) (m : Source.Memory α) :
    ∃ t,FP.Structural.exec M (Source.kernel .fp32 .fp32 (parameters r)) (state m row)=some t ∧
      t.mem=rowMemory M r row m :=
  ⟨_,(Source.execution M (parameters r) (state m row)).property,
    Source.execution_memory M (parameters r) (state m row)⟩

theorem active_iff (heads half : Nat) {H W : Nat} (i : TileIndex [H,W]) :
    Source.active heads half i ↔ i.1.val<heads ∧ i.2.1.val<half := by
  simp [Source.active,Bool.and_eq_true]
theorem half_column (r : Request) (second : Bool) (j : Nat) (hj : j<r.half) :
    j+(if second then r.half else 0)<r.dimension := by
  have : 2*r.half≤r.dimension := by unfold Request.half; omega
  cases second <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
theorem address_coordinate (r : Request) (key second : Bool) (row : Nat)
    {H W : Nat} (i : TileIndex [H,W]) :
    Source.address row (r.heads key*r.dimension) r.dimension r.half second i=
      r.offset key row i.1.val (i.2.1.val+if second then r.half else 0) := by
  simp only [Source.address,Request.offset]
  ring
theorem address_injective (r : Request) (key second : Bool) (row H W : Nat)
    (hw : W≤r.dimension) :
    Function.Injective (@Source.address row (r.heads key*r.dimension) r.dimension r.half second H W) := by
  intro i j same
  have hd : 0<r.dimension := by have := i.2.1.isLt; omega
  have hi : i.2.1.val<r.dimension := lt_of_lt_of_le i.2.1.isLt hw
  have hj : j.2.1.val<r.dimension := lt_of_lt_of_le j.2.1.isLt hw
  have eq : i.1.val*r.dimension+i.2.1.val=j.1.val*r.dimension+j.2.1.val := by
    unfold Source.address at same
    omega
  have lane := congrArg (fun n => n%r.dimension) eq
  simp [Nat.add_mod,Nat.mod_eq_of_lt hi,Nat.mod_eq_of_lt hj] at lane
  have div (a b : Nat) (hb : b<r.dimension) : (a*r.dimension+b)/r.dimension=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hd,Nat.div_eq_of_lt hb]; simp
  have head := congrArg (fun n => n/r.dimension) eq
  simp only [div _ _ hi,div _ _ hj] at head
  exact Prod.ext (Fin.ext head) (Prod.ext (Fin.ext lane) (Subsingleton.elim _ _))

theorem halves_apart (r : Request) (key side : Bool) (row : Nat)
    {H W H' W' : Nat} (i : TileIndex [H,W]) (j : TileIndex [H',W'])
    (hi : Source.active (r.heads key) r.half i) (hj : Source.active (r.heads key) r.half j) :
    Source.address row (r.heads key*r.dimension) r.dimension r.half side i ≠
      Source.address row (r.heads key*r.dimension) r.dimension r.half (!side) j := by
  intro eq
  obtain ⟨hiH,hiL⟩ := (active_iff _ _ _).mp hi
  obtain ⟨hjH,hjL⟩ := (active_iff _ _ _).mp hj
  rw [address_coordinate,address_coordinate] at eq
  have same := (offset_injective r key hiH (half_column r side _ hiL) hjH
    (half_column r (!side) _ hjL) eq).2.2
  cases side <;> simp only [Bool.not_false,Bool.not_true,Bool.false_eq_true,↓reduceIte] at same <;> omega

theorem scatter_readback {α : Type} [Inhabited α] (m : Source.Memory α) (target : RegionName)
    (pid stride heads hd : Nat) (second : Bool) (H W : Nat)
    (value : TileIndex [H,W]→Cell α) (i : TileIndex [H,W])
    (hi : Source.active heads (hd/2) i)
    (hinj : Function.Injective (@Source.address pid stride hd (hd/2) second H W)) :
    Source.scatter m target pid stride heads hd second H W value target
      (Source.address pid stride hd (hd/2) second i)=value i := by
  have result := State.masked_scatter_readback target (Source.address pid stride hd (hd/2) second)
    value (fun i => Source.active heads (hd/2) i) (state m pid) i hi hinj
  rw [fold_writes_mem] at result
  exact result

abbrev Lane (r : Request) (key : Bool) := TileIndex [r.paddedHeads key,r.paddedDim/2]
def address (r : Request) (row : Nat) (key side : Bool) (i : Lane r key) :=
  Source.address row (r.heads key*r.dimension) r.dimension r.half side i
def value {α : Type} [Inhabited α] (op : Operations α) (r : Request) (m : Source.Memory α)
    (row : Nat) (key side : Bool) (i : Lane r key) : α :=
  Source.laneResult op.backend .fp32 .fp32 (parameters r) (state m row) (region key)
    (r.heads key*r.dimension) (r.heads key) side i
def write {α : Type} [Inhabited α] (op : Operations α) (r : Request) (m : Source.Memory α)
    (row : Nat) (key side : Bool) (current : Source.Memory α) : Source.Memory α :=
  Source.scatter current (region key) row (r.heads key*r.dimension) (r.heads key) r.dimension
    side (r.paddedHeads key) (r.paddedDim/2) (fun i => .mk .fp32 (value op r m row key side i))

theorem row_writes {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (row : Nat) (m : Source.Memory α) :
    rowMemory op.backend r row m=
      write op r m row Bool.true Bool.true (write op r m row Bool.true Bool.false
        (write op r m row Bool.false Bool.true (write op r m row Bool.false Bool.false m))) := by
  rfl

theorem value_eq {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (row : Nat) (key side : Bool) (i : Lane r key)
    (hi : Source.active (r.heads key) r.half i) :
    value op r m row key side i=scalarPair op side
      ((m (region key) (address r row key Bool.false i)).read .fp32)
      ((m (region key) (address r row key Bool.true i)).read .fp32)
      ((m "C" (row%r.sequence*r.cosRowStride+i.2.1.val*r.cosColStride)).read .fp32)
      ((m "S" (row%r.sequence*r.sinRowStride+i.2.1.val*r.sinColStride)).read .fp32) := by
  have hj : i.2.1.val<r.dimension/2 := ((active_iff _ _ i).mp hi).2
  cases side <;>
    simp [value,Source.laneResult,Source.coefficient,Source.component,parameters,state,
      Request.half,hi,hj,Operations.backend,scalarPair,Source.readFloat,toFloat,
      FloatDType.toTileDType,address,address_coordinate] at *

theorem write_readback {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m current : Source.Memory α) (row : Nat) (key side : Bool) (i : Lane r key)
    (hi : Source.active (r.heads key) r.half i) :
    write op r m row key side current (region key) (address r row key side i)=
      .mk .fp32 (value op r m row key side i) := by
  unfold write address
  exact scatter_readback current _ _ _ _ _ _ _ _ _ i hi
    (address_injective r key side row _ _ (padded_half_le _))

theorem write_frame {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m current : Source.Memory α) (row : Nat) (key side : Bool) (name : RegionName) (o : Nat)
    (h : name≠region key ∨ ∀ i : Lane r key,Source.active (r.heads key) r.half i →
      o≠address r row key side i) :
    write op r m row key side current name o=current name o := by
  apply Source.scatter_frame
  exact h

theorem write_other_key {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m current : Source.Memory α) (row : Nat) (key side : Bool) (o : Nat) :
    write op r m row key side current (region (!key)) o=current (region (!key)) o := by
  apply write_frame
  left
  cases key <;> decide

theorem write_other_side {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m current : Source.Memory α) (row : Nat) (key side : Bool) (i : Lane r key)
    (hi : Source.active (r.heads key) r.half i) :
    write op r m row key side current (region key) (address r row key (!side) i)=
      current (region key) (address r row key (!side) i) := by
  apply write_frame
  right
  intro j hj
  simpa only [address,address_coordinate,Bool.not_not] using halves_apart r key (!side) row i j hi hj

theorem row_readback {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (m : Source.Memory α) (row : Nat) (key side : Bool) (i : Lane r key)
    (hi : Source.active (r.heads key) r.half i) :
    rowMemory op.backend r row m (region key) (address r row key side i)=
      .mk .fp32 (value op r m row key side i) := by
  rw [row_writes]
  have hk (side : Bool) (current : Source.Memory α) (o : Nat) :
      write op r m row Bool.true side current (region Bool.false) o=current (region Bool.false) o := by
    simpa only [Bool.not_true] using write_other_key op r m current row Bool.true side o
  have hs (key : Bool) (current : Source.Memory α) (i : Lane r key)
      (hi : Source.active (r.heads key) r.half i) :
      write op r m row key Bool.true current (region key) (address r row key Bool.false i)=
      current (region key) (address r row key Bool.false i) := by
    simpa only [Bool.not_true] using write_other_side op r m current row key Bool.true i hi
  cases key <;> cases side
  · rw [hk,hk,hs _ _ _ hi]
    exact write_readback _ _ _ _ _ _ _ _ hi
  · rw [hk,hk]
    exact write_readback _ _ _ _ _ _ _ _ hi
  · rw [hs _ _ _ hi]
    exact write_readback _ _ _ _ _ _ _ _ hi
  · exact write_readback _ _ _ _ _ _ _ _ hi

def OutsideRow (r : Request) (row : Nat) (name : RegionName) (o : Nat) : Prop :=
  ∀ key side,(name≠region key ∨ ∀ i : Lane r key,Source.active (r.heads key) r.half i → o≠address r row key side i)

theorem row_frame {α : Type} [Inhabited α] (M : Algebra α) (r : Request)
    (m : Source.Memory α) (row : Nat) (name : RegionName) (o : Nat)
    (h : OutsideRow r row name o) : rowMemory M r row m name o=m name o := by
  apply Source.reference_frame
  exact ⟨fun side => h Bool.false side,fun side => h Bool.true side⟩

#axiomsClean row_readback
#axiomsClean row_frame
#axiomsClean value_eq

#axiomsClean row_execution
#axiomsClean halves_apart
#axiomsClean scatter_readback
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
