import bench.repairs.tritonbench_g.triton_matmul.Proofs.ByteExecution
import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteHost
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.ByteHost
open VeriTile.Triton FP.Structural
open FPValue (Storage DotAcc)
open Contract TensorBinding InputAdapter OutputBytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000

abbrev Allocator := Tensor → Tensor → ByteExecution.Machine → Option (Tensor × ByteExecution.Machine)

/-- Ordinary fresh allocation: its own storage may change, inputs and exterior
bytes are preserved, and the source's C view denotes its physical words.
No execution result or source-specific arithmetic bound is a field. -/
structure Allocation (a b c : Tensor) (before prepared : ByteExecution.Machine) : Prop where
  ordinary : InputAdapter.Allocation a b c before.1 prepared.1
  tensor : Laws (storage a.metadata) c
  output : Bound prepared.2 (storage a.metadata) c prepared.1
  frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width (storage a.metadata)*c.metadata.numel≤ptr →
    prepared.2 ptr=before.2 ptr

def AllocationLaw (allocate : Allocator) (a b : Tensor) (before : ByteExecution.Machine) : Prop :=
  ∃ c prepared,allocate a b before=some (c,prepared) ∧ Allocation a b c before prepared

def Allocated (allocate : Allocator) (a b : Tensor) (before : ByteExecution.Machine) (c : Tensor) : Prop :=
  ∃ prepared,allocate a b before=some (c,prepared) ∧ Allocation a b c before prepared

def logicalAllocate (allocate : Allocator) (read : Read) : InputAdapter.Allocator Word :=
  fun a b s => (allocate a b (s,read)).map (fun (c,prepared) => (c,prepared.1))

def call (rest : Algebra Word) (I : Native.Instructions) (allocate : Allocator)
    (a b : Tensor) (before : ByteExecution.Machine) : Option (Tensor × ByteExecution.Machine) :=
  if checkTensorInputs a.metadata b.metadata=.ok () then
    (allocate a b before).bind fun (c,prepared) =>
      if Contract.Empty a.metadata b.metadata then some (c,prepared)
      else (ByteExecution.grid (Native.algebra rest I) (Native.dotAcc I) (request a b) c
        (List.range (programs (request a b))) prepared).map (fun final => (c,final))
  else none

structure Post (read final : Read) (a b c : Tensor) (lc : Laws (storage a.metadata) c)
    (value : OutputIndex a b → Word) : Prop where
  metadata : Metadata a b c
  words : ∀ i,readWord final (storage a.metadata)
    (c.metadata.dataPtr.toNat+width (storage a.metadata)*address a b i)=value i
  logical : ∀ i,logicalValue final (storage a.metadata) c lc (address a b i)=value i
  bytes : ∀ i j,j<width (storage a.metadata) →
    final (physicalAddress (storage a.metadata) c (address a b i)+j)=byte (value i) j
  frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width (storage a.metadata)*c.metadata.numel≤ptr →
    final ptr=read ptr

theorem post (read ready : Read) (a b c : Tensor) (lc : Laws (storage a.metadata) c)
    (initial final : State Word) (value : OutputIndex a b → Word)
    (logical : CheckedHost.Post a b c initial final value)
    (packed : ∀ i,pack (storage a.metadata) (value i)=value i)
    (frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width (storage a.metadata)*c.metadata.numel≤ptr →
      ready ptr=read ptr) : Post read (render ready (storage a.metadata) c final) a b c lc value := by
  have ib : ∀ i : OutputIndex a b,address a b i<c.metadata.numel := by
    intro i
    rw [output_numel a b c logical.metadata]
    exact linear_bound _ _ _ _ i.1.isLt i.2.isLt
  have ptr : ∀ i : OutputIndex a b,c.metadata.dataPtr.toNat+width (storage a.metadata)*address a b i=
      physicalAddress (storage a.metadata) c (address a b i) := by
    intro i
    rw [lc.pointer (by have := ib i; omega)]
    simp only [physicalAddress,lc.contiguous logical.metadata.2.2.2.1 _ (ib i),Nat.mul_add,Nat.add_assoc]
  have words : ∀ i : OutputIndex a b,readWord (render ready (storage a.metadata) c final) (storage a.metadata)
      (c.metadata.dataPtr.toNat+width (storage a.metadata)*address a b i)=value i := by
    intro i
    rw [render_read ready (storage a.metadata) c final _ (ib i),logical.values i,read_stored,packed i]
  refine ⟨logical.metadata,words,?_,?_,?_⟩
  · intro i
    rw [←direct_read (render ready (storage a.metadata) c final) _ c lc logical.metadata.2.2.2.1 logical.metadata.2.2.2.2 _ (ib i)]
    exact words i
  · intro i j hj
    rw [←ptr i,render_at ready (storage a.metadata) c final _ j (ib i) hj,logical.values i,read_stored]
  · intro addr outside
    exact (OutputBytes.frame ready (storage a.metadata) c final addr outside).trans (frame addr outside)

theorem expected_packed (I : Native.Instructions) (a b : Tensor)
    (A : Fin (rows a.metadata) → Fin (columns a.metadata) → Word)
    (B : Fin (columns a.metadata) → Fin (columns b.metadata) → Word) (i : OutputIndex a b) :
    pack (storage a.metadata) (logicalReference (Native.numerics I (storage a.metadata)) (request a b) A B i.1 i.2)=
      logicalReference (Native.numerics I (storage a.metadata)) (request a b) A B i.1 i.2 :=
  pack_native I (storage a.metadata) _

theorem correct (rest : Algebra Word) (I : Native.Instructions) (allocate : Allocator)
    (read : Read) (a b : Tensor) (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State Word)
    (allocates : AllocationLaw allocate a b (initial,read)) (bound : Inputs read id a b initial) :
    ∃ c final bytes, ∃ lc : Laws (storage a.metadata) c,
      Allocated allocate a b (initial,read) c ∧
      call rest I allocate a b (initial,read)=some (c,(final,bytes)) ∧
      CheckedHost.Post a b c initial final (fun i => logicalReference (Native.numerics I (storage a.metadata)) (request a b)
        (CheckedHost.left read id a b la) (CheckedHost.right read id a b lb) i.1 i.2) ∧
      Post read bytes a b c lc (fun i => logicalReference (Native.numerics I (storage a.metadata)) (request a b)
        (CheckedHost.left read id a b la) (CheckedHost.right read id a b lb) i.1 i.2) := by
  obtain ⟨c0,⟨prepared0,ready⟩,allocated,allocation⟩ := allocates
  have logicalLaw : InputAdapter.AllocationLaw (logicalAllocate allocate read) a b initial :=
    ⟨c0,prepared0,by simp only [logicalAllocate,allocated,Option.map_some],allocation.ordinary⟩
  obtain ⟨c,prepared,final,allocEq,called,logical⟩ := FiniteHost.correct (Native.algebra rest I) (Native.dotAcc I)
    (Native.numerics I (storage a.metadata)) (logicalAllocate allocate read) read id a b la lb h
    (Native.primitives rest I _) initial logicalLaw bound
  have eq : (c0,prepared0)=(c,prepared) := by
    simp only [logicalAllocate,allocated,Option.map_some] at allocEq
    exact Option.some.inj allocEq
  have ec := (Prod.mk.inj eq).1
  have ep := (Prod.mk.inj eq).2
  subst c0; subst prepared0
  have checked := (checkTensorInputs_iff a.metadata b.metadata).mpr h
  by_cases empty : Contract.Empty a.metadata b.metadata
  · have ef : prepared=final := by
      simp only [FiniteHost.call,if_pos checked,allocEq,Option.bind_some,if_pos empty] at called
      exact (Prod.mk.inj (Option.some.inj called)).2
    subst final
    refine ⟨c,prepared,ready,allocation.tensor,⟨(prepared,ready),allocated,allocation⟩,?_,logical,?_⟩
    · simp only [call,if_pos checked,allocated,Option.bind_some,if_pos empty]
    · have output := post read ready a b c allocation.tensor initial prepared _ logical
        (expected_packed I a b _ _) allocation.frame
      rw [render_initial ready (storage a.metadata) c prepared allocation.output] at output
      exact output
  · let tensor := tensorAPI read id a b c h empty initial prepared allocation.ordinary bound
    have hl : programs (request a b)<2^31 := (execution_nonempty _ _ h empty).1.2.2.2
    have base : tensor.baseC=c.metadata.dataPtr.toNat := by
      change InputAdapter.base c=c.metadata.dataPtr.toNat
      have size := output_numel a b c allocation.ordinary.metadata
      have nz : c.metadata.numel≠0 := by
        unfold Contract.Empty at empty
        rw [size]
        exact Nat.mul_ne_zero (by omega) (by omega)
      simp only [InputAdapter.base,if_neg nz]
    have seq := ByteExecution.grid_render (Native.algebra rest I) (Native.dotAcc I) (request a b) c
      prepared tensor hl rfl (output_numel a b c allocation.ordinary.metadata).symm base ready
      (List.range (programs (request a b))) (fun _ hp => List.mem_range.mp hp) prepared
    have storage_eq : (request a b).storage=storage a.metadata := rfl
    rw [storage_eq,render_initial ready (storage a.metadata) c prepared allocation.output] at seq
    have grid : FiniteGrid.grid (Native.algebra rest I) (Native.dotAcc I) (request a b)
        (List.range (programs (request a b))) prepared=some final := by
      simp only [FiniteHost.call,if_pos checked,allocEq,Option.bind_some,if_neg empty] at called
      obtain ⟨t,ht,he⟩ := Option.map_eq_some_iff.mp called
      have he := (Prod.mk.inj he).2
      simpa only [he] using ht
    refine ⟨c,final,render ready (storage a.metadata) c final,allocation.tensor,⟨(prepared,ready),allocated,allocation⟩,?_,logical,?_⟩
    · simp only [call,if_pos checked,allocated,Option.bind_some,if_neg empty,seq,grid,Option.map_some]
    · exact post read ready a b c allocation.tensor initial final _ logical (expected_packed I a b _ _) allocation.frame

theorem output_addresses (a b c : Tensor) (before prepared : ByteExecution.Machine)
    (allocation : Allocation a b c before prepared) (i : OutputIndex a b)
    (j : Nat) (hj : j<width (storage a.metadata)) :
    physicalAddress (storage a.metadata) c (address a b i)+j<2^64 := by
  have ib : address a b i<c.metadata.numel := by
    rw [output_numel a b c allocation.ordinary.metadata]
    exact linear_bound _ _ _ _ i.1.isLt i.2.isLt
  have cp : c.metadata.dataPtr.toNat+width (storage a.metadata)*address a b i=
      physicalAddress (storage a.metadata) c (address a b i) := by
    rw [allocation.tensor.pointer (by omega)]
    simp only [physicalAddress,allocation.tensor.contiguous allocation.ordinary.metadata.2.2.2.1 _ ib,
      Nat.mul_add,Nat.add_assoc]
  rw [←cp]
  have window := allocation.ordinary.window
  have bw := output_bytes a b c allocation.ordinary.metadata
  unfold Contract.Window at window
  rw [bw] at window
  rcases window with empty | ⟨_,limit⟩
  · omega
  · have bound := Nat.mul_le_mul_left (width (storage a.metadata)) (show address a b i+1≤c.metadata.numel by omega)
    rw [Nat.mul_add,Nat.mul_one] at bound
    omega

theorem separate_frame (read final : Read) (st : Storage) (input c : Tensor)
    (separate : MatmulTritonAutotune.Whole.byteSpansDisjoint c.metadata.dataPtr.toNat (width st*c.metadata.numel)
      input.metadata.dataPtr.toNat (width st*input.metadata.numel))
    (frame : ∀ ptr,ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width st*c.metadata.numel≤ptr → final ptr=read ptr)
    (i j : Nat) (hi : i < input.metadata.numel) (hj : j<width st) :
    final (input.metadata.dataPtr.toNat+width st*i+j)=read (input.metadata.dataPtr.toNat+width st*i+j) := by
  apply frame
  have bound := Nat.mul_le_mul_left (width st) (show i+1 ≤ input.metadata.numel by omega)
  rw [Nat.mul_add,Nat.mul_one] at bound
  have positive : 0<width st := by cases st <;> decide
  unfold MatmulTritonAutotune.Whole.byteSpansDisjoint at separate
  rcases separate with empty | empty | left | right <;> omega

theorem input_frame (allocate : Allocator) (a b c : Tensor) (before : State Word) (read bytes : Read)
    (allocated : Allocated allocate a b (before,read) c) (lc : Laws (storage a.metadata) c)
    (value : OutputIndex a b → Word) (post : Post read bytes a b c lc value) :
    (∀ i j,i<a.metadata.numel → j<width (storage a.metadata) →
      bytes (a.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)=read (a.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)) ∧
    (∀ i j,i<b.metadata.numel → j<width (storage a.metadata) →
      bytes (b.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)=read (b.metadata.dataPtr.toNat+width (storage a.metadata)*i+j)) := by
  obtain ⟨prepared,_,allocation⟩ := allocated
  exact ⟨separate_frame read bytes (storage a.metadata) a c allocation.ordinary.separateA post.frame,
    separate_frame read bytes (storage a.metadata) b c allocation.ordinary.separateB post.frame⟩

#axiomsClean output_addresses
#axiomsClean input_frame
#axiomsClean post
#axiomsClean expected_packed
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.ByteHost
