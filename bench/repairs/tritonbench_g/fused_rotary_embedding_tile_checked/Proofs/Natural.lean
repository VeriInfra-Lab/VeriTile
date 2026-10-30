import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Program
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Layout

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem quot_nat (a b : Nat) : quot (a : Int) b = (a/b : Nat) := by
  simp [quot,Int.natCast_ediv]
@[simp] theorem rem_nat (a b : Nat) : rem (a : Int) b = (a%b : Nat) := by
  simp only [rem,quot_nat,Int.natCast_ediv,Int.natCast_emod,Int.emod_def]
@[simp] theorem dimension_nat (s : Shape) (high : Bool) (i : Nat) :
    dimension (parameters s) high i = (i+if high then s.half else 0 : Nat) := by
  cases high <;> simp [dimension,parameters,quot,Shape.half]
@[simp] theorem active_nat (s : Shape) (i : HalfLane (parameters s)) :
    active (parameters s) i ↔ i.1.val<s.half := by
  have nonnegative : ¬(s.width : Int)<0 := by omega
  simp [active,parameters,quot,Shape.half,nonnegative]
  omega
@[simp] theorem head_nat (s : Shape) (pid : Program) (m : Mem α) (key : Bool) :
    head (parameters s) (state pid m) key = (if key then pid.1/s.groups else pid.1 : Nat) := by
  cases key <;> simp [head,state,parameters]

def work (s : Shape) (pid : Program) (key : Bool) (i : Nat) : Work :=
  ⟨if key then .k else .q,pid.2,if key then pid.1/s.groups else pid.1,i⟩

@[simp] theorem dataOffset_nat (s : Shape) (pid : Program) (m : Mem α)
    (key high : Bool) (i : HalfLane (parameters s)) :
    dataOffset (parameters s) (state pid m) key high i = s.input (work s pid key i.1.val) high := by
  simp only [dataOffset,head_nat,dimension_nat]
  cases key <;> cases high <;>
    simp only [parameters,state,Bool.false_eq_true,ite_false,ite_true,Nat.cast_one,
      Int.mul_one,Shape.input,Shape.data,Shape.heads,Shape.dim,work,Shape.half,
      Nat.cast_zero,Int.add_zero,Nat.add_zero,Kind.noConfusion,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast] <;> simp <;> ring


@[simp] theorem coefficientOffset_nat (s : Shape) (pid : Program) (m : Mem α)
    (sine : Bool) (i : HalfLane (parameters s)) :
    coefficientOffset (parameters s) (state pid m) sine i = s.coeff sine pid.2 i.1.val := by
  cases sine <;> simp only [coefficientOffset,state,parameters,Shape.coeff,
    Bool.false_eq_true,ite_true,ite_false,Int.mul_one,Nat.cast_one,
    ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast] <;> simp

@[simp] theorem past_nat (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (m : Memory α) :
    past (parameters c.shape) (state pid (lift c m)) = (c.past pid.2 : Int) := by
  have positive:=hc.lengthPositive pid.2 ht
  simp [past,parameters,state,lift,Cell.read,Config.past]
  omega
@[simp] theorem block_nat (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (m : Memory α) :
    block (parameters c.shape) (state pid (lift c m)) = (c.block pid.2 : Int) := by
  unfold block
  rw [past_nat c hc pid ht m]
  simp only [parameters,quot_nat,state,lift,Cell.read,Config.block,
    Nat.cast_one,Int.mul_one,Nat.one_ne_zero,ite_false,ite_true,
    ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp
  simp only [←Int.natCast_ediv,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
@[simp] theorem slot_nat (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (m : Memory α) :
    slot (parameters c.shape) (state pid (lift c m)) = (c.slot pid.2 : Int) := by
  unfold slot
  rw [past_nat c hc pid ht m]
  exact rem_nat _ _

@[simp] theorem cacheOffset_nat (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (m : Memory α) (high : Bool)
    (i : HalfLane (parameters c.shape)) (hi : i.1.val<c.shape.half) :
    cacheOffset (parameters c.shape) (state pid (lift c m)) high i =
      c.output (work c.shape pid Bool.true i.1.val) high := by
  have dimBound : i.1.val+(if high then c.shape.half else 0)<c.shape.width := by
    have := width_halves c.shape hc.shape
    cases high <;> simp <;> omega
  simp only [cacheOffset,block_nat c hc pid ht m,head_nat,slot_nat c hc pid ht m,dimension_nat]
  cases layout : c.shape.newLayout <;>
    simp only [parameters,layout,Bool.false_eq_true,ite_false,ite_true,
      quot_nat,rem_nat,Config.output,work,Shape.cache,Shape.dim,Kind.noConfusion,
      Bool.and_self,Nat.cast_zero,Nat.cast_one,Int.mul_zero,Int.mul_one,Int.add_zero,
      Nat.mod_eq_of_lt dimBound,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast] <;> simp <;> ring

@[simp] theorem valueOffset_nat (s : Shape) (pid : Program) (m : Mem α)
    (i : FullLane (parameters s)) :
    valueOffset (parameters s) (state pid m) i = s.data .v pid.2 (pid.1/s.groups) i.1.val := by
  simp only [valueOffset,head_nat]
  simp only [state,parameters,Shape.data,Shape.heads,Kind.noConfusion,ite_false,ite_true,
    Nat.cast_one,Int.mul_one,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp
  ring
@[simp] theorem valueCacheOffset_nat (c : Config) (hc : c.Valid) (pid : Program)
    (ht : pid.2<c.shape.tokens) (m : Memory α) (i : FullLane (parameters c.shape)) :
    valueCacheOffset (parameters c.shape) (state pid (lift c m)) i =
      c.shape.cache Bool.false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) i.1.val := by
  simp only [valueCacheOffset,block_nat c hc pid ht m,head_nat,slot_nat c hc pid ht m]
  simp only [parameters,Shape.cache,Bool.false_and,Bool.false_eq_true,ite_false,ite_true,
    Nat.cast_one,Int.mul_one,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp
  ring

@[simp] theorem pair_nat (rest : Algebra α) (ops : Ops α) (c : Config)
    (pid : Program) (m : Memory α) (key high : Bool) (i : HalfLane (parameters c.shape))
    (hi : i.1.val<c.shape.half) :
    pair rest ops (parameters c.shape) (state pid (lift c m)) key high i =
      .mk .real (component (pairValues c.shape (fun _=>ops) (work c.shape pid key i.1.val) m) high) := by
  simp only [pair,load,active_nat,hi,ite_true,dataOffset_nat,coefficientOffset_nat]
  cases key <;> simp [state,lift,decoded,Cell.read,pairValues,work,Kind.input]

#axiomsClean cacheOffset_nat
#axiomsClean pair_nat
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
