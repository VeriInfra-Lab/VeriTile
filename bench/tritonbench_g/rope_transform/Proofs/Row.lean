import bench.tritonbench_g.rope_transform.Proofs.Index
namespace VeriTile.Bench.Repairs.RopeTransform
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Independent pair formula with the exact source scalar operation boundaries.
The inputs are the launch snapshot, never a previous store result. -/
def scalar {α : Type} (M : Algebra α) (x y c sn : α) (second : Bool) : α :=
  let a := M.cast none .fp32 .fp32 x
  let b := M.cast none .fp32 .fp32 y
  M.cast none .fp32 .fp32
    (if second then M.binary none .fp32 .add (M.binary none .fp32 .mul b c) (M.binary none .fp32 .mul a sn)
     else M.binary none .fp32 .sub (M.binary none .fp32 .mul a c) (M.binary none .fp32 .mul b sn))

def value {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (key : Bool) (a h : Nat) (second : Bool) (j : Nat) : α :=
  scalar M ((s.mem (region key) (r.offset key a h j)).read .fp32)
    ((s.mem (region key) (r.offset key a h (j+r.half))).read .fp32)
    ((s.mem "C" (a%r.sequence*r.half+j)).read .fp32)
    ((s.mem "S" (a%r.sequence*r.half+j)).read .fp32) second

def write {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (key second : Bool) (mem : T.Memory α) : T.Memory α :=
  T.scatter mem (region key) (s.pids 0) (r.stride key) (r.heads key) r.dimension
    second (r.paddedHeads key) r.width
    (fun i => Cell.mk .fp32 (T.laneResult M .fp32 .fp32 r.parameters s (region key)
      (r.stride key) (r.heads key) second i))

theorem reference_eq {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α) :
    T.referenceMemory M .fp32 .fp32 r.parameters s=
      write M r s Bool.true Bool.true (write M r s Bool.true Bool.false
        (write M r s Bool.false Bool.true (write M r s Bool.false Bool.false s.mem))) := rfl

theorem scalar_lane {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (key second : Bool) (h j : Nat) (hh : h<r.heads key) (hj : j<r.half) :
    T.laneResult M .fp32 .fp32 r.parameters s (region key) (r.stride key) (r.heads key)
      second (lane r key h j hh hj)=value M r s key (s.pids 0) h second j := by
  cases second <;>
    simp [T.laneResult,T.component,T.coefficient,T.readFloat,T.active,T.address,
      Request.parameters,half_eq,lane,hh,hj,scalar,value,Request.offset,
      toFloat,Nat.add_assoc]

theorem scatter_at {α : Type} (r : Request) (s : State α) (key second : Bool)
    (v : TileIndex [r.paddedHeads key,r.width] → Cell α)
    (i : TileIndex [r.paddedHeads key,r.width])
    (hi : T.active (r.heads key) (r.dimension/2) i) :
    T.scatter s.mem (region key) (s.pids 0) (r.stride key) (r.heads key) r.dimension second
      (r.paddedHeads key) r.width v (region key)
        (T.address (s.pids 0) (r.stride key) r.dimension (r.dimension/2) second i)=v i := by
  have h := State.masked_scatter_readback (region key)
    (T.address (s.pids 0) (r.stride key) r.dimension (r.dimension/2) second) v
    (fun i => T.active (r.heads key) (r.dimension/2) i) s i hi
    (address_injective r key (s.pids 0) second)
  rw [fold_writes_mem] at h
  exact h

theorem write_at {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (mem : T.Memory α) (key second : Bool) (h j : Nat)
    (hh : h<r.heads key) (hj : j<r.half) :
    write M r s key second mem (region key) (r.offset key (s.pids 0) h (r.column second j))=
      Cell.mk .fp32 (value M r s key (s.pids 0) h second j) := by
  have hx := scatter_at r {s with mem:=mem} key second
    (fun i => Cell.mk .fp32 (T.laneResult M .fp32 .fp32 r.parameters s (region key)
      (r.stride key) (r.heads key) second i))
    (lane r key h j hh hj) (lane_active r key h j hh hj)
  rw [address_eq] at hx
  change write M r s key second mem (region key) (r.offset key (s.pids 0) h (r.column second j))=
    Cell.mk .fp32 (T.laneResult M .fp32 .fp32 r.parameters s (region key)
      (r.stride key) (r.heads key) second (lane r key h j hh hj)) at hx
  rw [scalar_lane] at hx
  exact hx

theorem write_other {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (mem : T.Memory α) (key second target part : Bool) (h j : Nat)
    (hh : h<r.heads target) (hj : j<r.half)
    (apart : key≠target ∨ second≠part) :
    write M r s key second mem (region target) (r.offset target (s.pids 0) h (r.column part j))=
      mem (region target) (r.offset target (s.pids 0) h (r.column part j)) := by
  apply T.scatter_frame
  by_cases hk : key=target
  · subst target
    right
    intro i hi he
    have hh' : i.1.val<r.heads key := (show i.1.val<r.heads key ∧ i.2.1.val<r.half from by simpa [T.active,half_eq] using hi).1
    have hj' : i.2.1.val<r.half := (show i.1.val<r.heads key ∧ i.2.1.val<r.half from by simpa [T.active,half_eq] using hi).2
    rw [address_eq] at he
    have hc := (offset_injective r key _ _ _ _ _ _ hh hh'
      (column_lt r part j hj) (column_lt r second _ hj') he).2.2
    have hs : second≠part := apart.resolve_left (fun h => h rfl)
    cases second <;> cases part <;> simp_all [Request.column]
    all_goals omega
  · left
    cases key <;> cases target <;> simp_all [region]

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (key second : Bool) (h j : Nat) (hh : h<r.heads key) (hj : j<r.half) :
    (T.execution M r.parameters s).val.mem (region key)
      (r.offset key (s.pids 0) h (r.column second j))=
        Cell.mk .fp32 (value M r s key (s.pids 0) h second j) := by
  rw [T.execution_memory,reference_eq]
  cases key <;> cases second
  · rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inl (by decide))]
    rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inl (by decide))]
    rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inr (by decide))]
    exact write_at M r s _ _ _ h j hh hj 
  · rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inl (by decide))]
    rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inl (by decide))]
    exact write_at M r s _ _ _ h j hh hj
  · rw [write_other _ _ _ _ _ _ _ _ _ _ hh hj (Or.inr (by decide))]
    exact write_at M r s _ _ _ h j hh hj
  · exact write_at M r s _ _ _ h j hh hj

def RowOutside (r : Request) (a : Nat) (name : RegionName) (o : Nat) : Prop :=
  ∀ key, name≠region key ∨ o<a*r.stride key ∨ (a+1)*r.stride key≤o

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (name : RegionName) (o : Nat) (out : RowOutside r (s.pids 0) name o) :
    (T.execution M r.parameters s).val.mem name o=s.mem name o := by
  rw [T.execution_memory]
  apply T.reference_frame
  have each : ∀ key second, name≠region key ∨
      ∀ i : TileIndex [r.paddedHeads key,r.width], T.active (r.heads key) (r.dimension/2) i →
        o≠T.address (s.pids 0) (r.stride key) r.dimension (r.dimension/2) second i := by
    intro key second
    rcases out key with hn | ho
    · exact Or.inl hn
    · right
      intro i hi he
      have hh : i.1.val<r.heads key := (show i.1.val<r.heads key ∧ i.2.1.val<r.half from by simpa [T.active,half_eq] using hi).1
      have hj : i.2.1.val<r.half := (show i.1.val<r.heads key ∧ i.2.1.val<r.half from by simpa [T.active,half_eq] using hi).2
      have hw := offset_window r key (s.pids 0) _ _ hh (column_lt r second _ hj)
      rw [address_eq] at he
      rcases ho with ho | ho <;> omega
  exact ⟨each Bool.false,each Bool.true⟩

#axiomsClean readback
#axiomsClean frame
#specNonCircular scalar avoiding [kernel,FP.Structural.exec,T.referenceMemory,T.laneResult]
end
end VeriTile.Bench.Repairs.RopeTransform
