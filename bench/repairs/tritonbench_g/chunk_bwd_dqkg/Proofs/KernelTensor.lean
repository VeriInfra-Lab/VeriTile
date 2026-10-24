import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelContract

/-! One shared Tensor placement binds actual pointer/storage metadata to both
the source Real interpretation and the native FP32 interpretation. The finite
commit is exactly the set of source stores over the whole grid. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelTensor
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32 Whole KernelContract
open VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit complete commit_output commit_frame native_logical_output)
end Shared
noncomputable section
open Classical
set_option maxHeartbeats 6000000
structure Buffers (c : Config) where
  tensor : RegionName → Tensor
  laws : ∀ r,TensorLaws (tensor r)
  checked : ∀ r,VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract (tensor r).metadata Bool.true
  strides : Role → List Nat
  isConj : Role → Bool
  admitted : Admitted c (fun r=>⟨(tensor (region r)).metadata,strides r,isConj r⟩)
def Buffers.view (b : Buffers c) (r : RegionName) : DataView :=
  dataView (b.tensor r) (b.laws r) ((b.checked r).dense rfl)
theorem Buffers.count (b : Buffers c) (r : Role) : (b.tensor (region r)).metadata.numel=size c r := by
  unfold VeriTile.Torch.TensorMetadata.numel size
  rw [(b.admitted.tensors r).2.2.2.2.2.2.2.1]
theorem size_extent (c : Config) (r : Role) : size c r=extent c (region r) := by
  cases r <;> simp [size,dimensions,region,extent,Config.heads,Nat.mul_assoc]
theorem Buffers.pointer (b : Buffers c) (r : RegionName) (hn : 0<(b.tensor r).metadata.numel) :
    BitVec.ofNat 64 ((b.view r).address 0)=(b.tensor r).dataPtr := actual_pointer _ _ _ hn
theorem Buffers.base (b : Buffers c) (r : RegionName) (hn : 0<(b.tensor r).metadata.numel) :
    (b.view r).base=(b.tensor r).metadata.dataPtr.toNat := by
  have eq := congrArg BitVec.toNat (b.pointer r hn)
  have span := (b.view r).spanFits
  have size : (b.view r).elements=(b.tensor r).metadata.numel := rfl
  have bound : (b.view r).base<2^64 := by rw [size] at span; omega
  simpa only [DataView.address,Nat.mul_zero,Nat.add_zero,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using eq
theorem logical_address (b : Buffers c) (r : RegionName) (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    (b.tensor r).storage.address ((b.tensor r).physicalIndex i)=(b.view r).address i := by
  rw [(b.laws r).contiguous ((b.checked r).dense rfl) i hi]
  simp only [Buffers.view,dataView,DataView.address,DataView.base,
    if_neg (by omega : (b.tensor r).metadata.numel≠0),Storage.address]
  omega

def Writes (c : Config) (r : RegionName) (i : Nat) : Prop := gridWrites c (allCTAs c) (r,i)
def outputRoles : List Role := [.dq,.dk,.dg]
theorem output_bounds (c : Config) (r : RegionName) (i : Nat) (active : Writes c r i) :
    ∃ q∈outputRoles,region q=r ∧ i<size c q := by
  obtain ⟨cta,_,ha⟩ := active
  rcases (writes_cases c cta r i).mp ha with ⟨rfl,row,e,h,eq⟩|⟨rfl,row,e,h,eq⟩|⟨rfl,row,h,eq⟩
  · exact ⟨.dq,by simp [outputRoles],rfl,by simpa [size_extent,extent,eq] using q_address_bound c cta row e h⟩
  · exact ⟨.dk,by simp [outputRoles],rfl,by simpa [size_extent,extent,eq] using q_address_bound c cta row e h⟩
  · exact ⟨.dg,by simp [outputRoles],rfl,by simpa [size_extent,extent,eq] using g_address_bound c cta row h⟩
theorem writes_nonempty (c : Config) (r : RegionName) (i : Nat) (active : Writes c r i) : nonempty c := by
  obtain ⟨cta,_,_⟩ := active
  have head := cta.head.isLt
  have ht := T_pos c cta
  have hp : 0<c.B*c.H := by simpa [Config.heads] using Nat.zero_lt_of_lt head
  have hb : 0<c.B := by
    by_contra h
    have z : c.B=0 := by omega
    simp [z] at hp
  have hh : 0<c.H := by
    by_contra h
    have z : c.H=0 := by omega
    simp [z] at hp
  exact ⟨hb,hh,ht⟩
def cells (c : Config) : List (RegionName × Nat) :=
  ((outputRoles.flatMap fun r=>(List.range (size c r)).map fun i=>(region r,i)).filter
    fun a=>decide (Writes c a.1 a.2)).dedup
theorem cells_coverage (c : Config) (r : RegionName) (i : Nat) : (r,i)∈cells c ↔ Writes c r i := by
  simp only [cells,List.mem_dedup,List.mem_filter,decide_eq_true_eq]
  constructor
  · exact And.right
  · intro active
    refine ⟨?_,active⟩
    obtain ⟨q,hq,eq,bound⟩ := output_bounds c r i active
    exact List.mem_flatMap.mpr ⟨q,hq,List.mem_map.mpr ⟨i,List.mem_range.mpr bound,by simp [eq]⟩⟩
theorem output_bound (b : Buffers c) (r : RegionName) (i : Nat) (active : Writes c r i) :
    i<(b.tensor r).metadata.numel := by
  obtain ⟨q,_,eq,h⟩ := output_bounds c r i active
  rw [←eq,b.count]
  exact h
theorem roles_apart (b : Buffers c) (r s : Role) (hn : nonempty c) (different : r≠s)
    (i j : Nat) (hi : i<(b.tensor (region r)).metadata.numel) (hj : j<(b.tensor (region s)).metadata.numel) :
    Memory.Apart ((b.view (region r)).address i) ((b.view (region s)).address j) := by
  have sep := b.admitted.separated r s hn different
  dsimp only at sep
  rw [←b.count r,←b.count s] at sep
  simp only [DataView.address,b.base _ (by omega : 0<(b.tensor (region r)).metadata.numel),
    b.base _ (by omega : 0<(b.tensor (region s)).metadata.numel),Memory.Apart]
  omega
theorem cells_apart (b : Buffers c) (a z : RegionName × Nat) (ha : a∈cells c) (hz : z∈cells c) (ne : a≠z) :
    Memory.Apart ((b.view a.1).address a.2) ((b.view z.1).address z.2) := by
  have aa := (cells_coverage c a.1 a.2).mp ha
  have zz := (cells_coverage c z.1 z.2).mp hz
  by_cases same : a.1=z.1
  · have neq : a.2≠z.2 := by intro eq; exact ne (Prod.ext same eq)
    simp only [DataView.address,same,Memory.Apart]
    omega
  · obtain ⟨r,_,ra,_⟩ := output_bounds c a.1 a.2 aa
    obtain ⟨s,_,rz,_⟩ := output_bounds c z.1 z.2 zz
    have different : r≠s := by intro eq; subst s; exact same (ra.symm.trans rz)
    rw [←ra,←rz]
    exact roles_apart b r s (writes_nonempty c _ _ aa) different a.2 z.2
      (by rw [ra]; exact output_bound b _ _ aa) (by rw [rz]; exact output_bound b _ _ zz)
def placement (b : Buffers c) : Shared.Placement (Writes c) where
  tensor:=b.tensor
  laws:=b.laws
  checked:=b.checked
  cells:=cells c
  unique:=List.nodup_dedup _
  coverage:=cells_coverage c
  bounds:=output_bound b
  apart:=fun a ha z hz ne=>cells_apart b a z ha hz ne

/-- The .real compatibility cells carry FP32 words in the native model.
The source's q/k/v/do loads instead use its .fp32 tag. -/
def floatCell {α : Type} (r : RegionName) (x : α) : Cell α :=
  if r="q" ∨ r="k" ∨ r="v" ∨ r="do" ∨ r="dq" ∨ r="dk" then .mk .fp32 x else .mk .real x
def physical {α : Type} [Inhabited α] (b : Buffers c) (read : RegionName → Nat → Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i=>if i<(b.tensor r).metadata.numel then floatCell r (read r i ((b.view r).address i)) else u.mem r i}
def logical {α : Type} [Inhabited α] (b : Buffers c) (read : RegionName → Nat → Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i=>if i<(b.tensor r).metadata.numel then
    floatCell r (read r i ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))) else u.mem r i}
theorem physical_eq_logical {α : Type} [Inhabited α] (b : Buffers c) (read : RegionName → Nat → Nat → α) (u : State α) :
    physical b read u=logical b read u := by
  have h : (physical b read u).mem=(logical b read u).mem := by
    funext r i
    by_cases hi : i<(b.tensor r).metadata.numel
    · simp [physical,logical,hi,logical_address b r i hi]
    · simp [physical,logical,hi]
  exact congrArg (fun m=>({u with mem:=m} : State α)) h

#axiomsClean physical_eq_logical
#axiomsClean cells_coverage
#axiomsClean cells_apart
#specNonCircular Buffers avoiding [Whole.runGrid,VeriTile.Triton.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelTensor
