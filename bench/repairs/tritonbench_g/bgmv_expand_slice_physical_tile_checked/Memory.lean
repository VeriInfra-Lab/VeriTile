import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Tensor
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.Bounds
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton Contract Direct
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Program (r : Request) (heap : ByteMemory) (h : Admitted r heap) :=
  Geometry.Program (call r heap h)

def outputIndex (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (p : Program r heap h) := p.batch*columns r.y+p.row+r.scalars.offset.toNat

theorem output_inside (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (p : Program r heap h) : outputIndex r heap h p<r.y.numel := by
  have c := checked r heap h
  have bound := Geometry.active_output (call r heap h) p
  have cap := Nat.mul_le_mul_right (columns r.y) c.shapes.2.2.2.1
  rw [Bounds.matrix_numel r.y c.ranks.2.1]
  exact bound.trans_le cap

structure Lane (r : Request) (heap : ByteMemory) (h : Admitted r heap) where
  program : Program r heap h
  active : indices r heap program.batch≠-1
  byte : Fin 4

def address (r : Request) (heap : ByteMemory) (h : Admitted r heap) (i : Lane r heap h) :=
  r.y.dataPtr.toNat+4*outputIndex r heap h i.program+i.byte.val

theorem outputIndex_unique (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (p q : Program r heap h) (same : outputIndex r heap h p=outputIndex r heap h q) : p=q := by
  have eq := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry.output_injective
    (call r heap h) p q (by simpa [outputIndex,Geometry.outputOffset,VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry.outputOffset,call,Contract.call] using same)
  cases p;cases q
  simp_all

theorem address_unique (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (i j : Lane r heap h) (same : address r heap h i=address r heap h j) : i=j := by
  have bi:=i.byte.isLt
  have bj:=j.byte.isLt
  simp only [address] at same
  have pe := outputIndex_unique r heap h i.program j.program (by omega)
  have be : i.byte=j.byte := Fin.ext (by omega)
  cases i;cases j
  simp_all

def writeback {α : Type} (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (values : Lane r heap h → α) (before : Nat → α) (a : Nat) : α :=
  if hit : ∃ i : Lane r heap h,address r heap h i=a then values (Classical.choose hit) else before a

theorem writeback_readback {α : Type} (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (values : Lane r heap h → α) (before : Nat → α) (i : Lane r heap h) :
    writeback r heap h values before (address r heap h i)=values i := by
  have hit : ∃ j : Lane r heap h,address r heap h j=address r heap h i := ⟨i,rfl⟩
  rw [writeback,dif_pos hit,address_unique r heap h _ i (Classical.choose_spec hit)]

theorem writeback_frame {α : Type} (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (values : Lane r heap h → α) (before : Nat → α) (a : Nat)
    (outside : ¬∃ i : Lane r heap h,address r heap h i=a) :
    writeback r heap h values before a=before a := by rw [writeback,dif_neg outside]

def ReadFrame {α : Type} (r : Request) (before after : Nat → α) : Prop :=
  ∀ t∈[r.x,r.w,r.i],∀ a,t.storageBase≤a → a<t.storageBase+t.storageBytes → after a=before a

theorem separate_excludes (x y : TensorInfo) (sep : ByteSeparate x y)
    (a : Nat) (lx : x.storageBase≤a) (ux : a<x.storageBase+x.storageBytes)
    (ly : y.storageBase≤a) (uy : a<y.storageBase+y.storageBytes) : False := by
  rcases sep with empty|empty|left|right <;> omega

theorem writeback_inputs {α : Type} (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (values : Lane r heap h → α) (before : Nat → α) :
    ReadFrame r before (writeback r heap h values before) := by
  intro t ht a low high
  apply writeback_frame
  rintro ⟨i,hit⟩
  have c := checked r heap h
  have span := fp32_span r.output c.dense.2.2.1.1 _ (output_inside r heap h i.program) i.byte
  change r.y.storageBase≤address r heap h i ∧ address r heap h i<r.y.storageBase+r.y.storageBytes at span
  rw [hit] at span
  have sep : ByteSeparate r.y t := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl
    · exact c.separate.1
    · exact c.separate.2.1
    · exact c.separate.2.2
  exact separate_excludes _ _ sep a span.1 span.2 low high

#axiomsClean output_inside
#axiomsClean address_unique
#axiomsClean writeback_readback
#axiomsClean writeback_frame
#axiomsClean writeback_inputs
#specNonCircular writeback avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
