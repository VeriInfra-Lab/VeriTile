import bench.tritonbench_g.streamk_matmul.Proofs.Half
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Native
open VeriTile.Triton
open Source
set_option maxHeartbeats 12000000
noncomputable section
abbrev Word := FP.Value .fp32
abbrev Half := BitVec 16
abbrev MMA8 := (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

/-- Actual ISA-local operations. MMA shape and input mode are fixed by the
source-specific SM80 lowering: m16n8k8.row.col.f32.tf32.tf32.f32. -/
structure Instructions where
  zero : Word
  mma8 : MMA8
  toHalf : Word → Half
  fromHalf : Half → Word
  atomicAdd : Word → Word → Word

/-- One fixed MMA instruction reference; its TF32 conversion, lane products,
seed, and exceptional-value behavior are the sole remaining MMA numerical
family. It cannot inspect program, tile, grid, memory, or a whole K extent. -/
structure Reference where
  mma8 : MMA8

def atomicConfig : FP.Config := ⟨.nearestEven,Bool.true,Bool.true⟩

def referenceInstructions (ref : Reference) : Instructions :=
  ⟨FP.Value.zero .fp32,ref.mma8,Whole001.Half.narrow,Whole001.Half.widen,FP.add atomicConfig⟩

def operations (actual : Instructions) : Source.Instructions Word :=
  ⟨actual.zero,actual.mma8,fun x=>actual.fromHalf (actual.toHalf x),actual.atomicAdd⟩

/-- Only local numerical equations. No schedule/publication/load/memory or
whole-output statement can be registered through this structure. -/
structure Numerical (actual : Instructions) (ref : Reference) : Prop where
  zero : actual.zero=FP.Value.zero .fp32
  mma8 : ∀ a b seed,actual.mma8 a b seed=ref.mma8 a b seed
  toHalf : ∀ x,actual.toHalf x=Whole001.Half.narrow x
  fromHalf : ∀ x,actual.fromHalf x=Whole001.Half.widen x
  atomicAdd : ∀ x y,actual.atomicAdd x y=FP.add atomicConfig x y

theorem numerical_consistent (ref : Reference) : Numerical (referenceInstructions ref) ref :=
  ⟨rfl,fun _ _ _=>rfl,fun _=>rfl,fun _=>rfl,fun _ _=>rfl⟩

theorem operations_eq (actual : Instructions) (ref : Reference) (h : Numerical actual ref) :
    operations actual=operations (referenceInstructions ref) := by
  have hm:actual.mma8=ref.mma8:=by funext a b seed; exact h.mma8 a b seed
  have hh:actual.toHalf=Whole001.Half.narrow:=by funext x; exact h.toHalf x
  have hf:actual.fromHalf=Whole001.Half.widen:=by funext x; exact h.fromHalf x
  have ha:actual.atomicAdd=FP.add atomicConfig:=by funext x y; exact h.atomicAdd x y
  simp only [operations,referenceInstructions,h.zero,hm,hh,hf,ha]

/-- Fixed m16n8k8 coordinate routing. This contains only finite integer
coordinates; physical data movement is the ordinary compiler boundary. -/
def aCoordinate (lane : Fin 32) (register : Fin 4) : Nat × Nat :=
  (lane.val/4+8*(register.val%2),lane.val%4+4*(register.val/2))
def bCoordinate (lane : Fin 32) (register : Fin 2) : Nat × Nat :=
  (lane.val%4+4*register.val,lane.val/4)
def cCoordinate (lane : Fin 32) (register : Fin 4) : Nat × Nat :=
  (lane.val/4+8*(register.val/2),2*(lane.val%4)+register.val%2)

theorem packed_bounds (lane : Fin 32) (a : Fin 4) (b : Fin 2) :
    (aCoordinate lane a).1 < 16 ∧ (aCoordinate lane a).2 < 8 ∧
    (bCoordinate lane b).1 < 8 ∧ (bCoordinate lane b).2 < 8 ∧
    (cCoordinate lane a).1 < 16 ∧ (cCoordinate lane a).2 < 8 := by
  have hl:=lane.isLt;have ha:=a.isLt;have hb:=b.isLt
  simp only [aCoordinate,bCoordinate,cCoordinate]
  omega

theorem output_owner (row : Fin 16) (col : Fin 8) :
    let lane : Fin 32 := ⟨row.val%8*4+col.val/2,by omega⟩
    let reg : Fin 4 := ⟨row.val/8*2+col.val%2,by omega⟩
    cCoordinate lane reg=(row.val,col.val) := by
  have hr:=row.isLt;have hc:=col.isLt
  simp only [cCoordinate]
  congr 1 <;> omega

theorem block_groups_complete (p : Parameters) (k : Fin p.h.bk) :
    k.val/8 < (p.h.bk+7)/8 ∧ k.val%8 < 8 ∧ k.val/8*8+k.val%8=k.val := by
  have he:=Arithmetic.native_block_length p
  have hm:=Nat.mod_lt k.val (by decide : 0 < 8)
  have hd:=Nat.div_add_mod k.val 8
  have hk:=k.isLt
  omega

theorem groups_unique (g h : Nat) (i j : Fin 8) (he : g*8+i.val=h*8+j.val) : g=h ∧ i=j := by
  have hi:=i.isLt;have hj:=j.isLt
  have hg:g=h:=by omega
  refine ⟨hg,Fin.ext ?_⟩
  omega

/-- Public word-level returned value, with no aggregate dot or output oracle.
Source memory, publication, arithmetic schedule and frames are already closed. -/
theorem returned_fp (p : Parameters) (actual : Instructions) (ref : Reference) (numeric : Numerical actual ref)
    (initial : Source.State Word p exterior) (cell : Returned.Cell p) :
    Returned.read p (Returned.tensor p (operations actual) initial) cell=
      Returned.value p (operations (referenceInstructions ref)) initial.inputA initial.inputB cell := by
  rw [Returned.value_contract,operations_eq actual ref numeric]

#axiomsClean numerical_consistent
#axiomsClean operations_eq
#axiomsClean packed_bounds
#axiomsClean output_owner
#axiomsClean block_groups_complete
#axiomsClean groups_unique
#axiomsClean returned_fp
#specNonCircular Numerical avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Returned.tensor,Source.chunkValue]
end
end Native
end VeriTile.Bench.Repairs.StreamK.Whole001
