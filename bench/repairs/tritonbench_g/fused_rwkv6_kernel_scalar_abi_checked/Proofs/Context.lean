import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Execution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Input-array recurrence. Its seed preserves 0+H0 and its padded key/value
lanes remain present; no output buffer or source evaluator is consulted. -/
def carry (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    Values α .real [c.BV,c.BK] :=
  fun i => scalarState (ops M) c (inputs c (memory s)) (s.pids 2)
    (s.pids 1*c.BK+i.2.1.val) (s.pids 0*c.BV+i.1.val) n

def terms (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat) :
    Values α .real [c.BV,c.BK] :=
  fun i => scalarTerm (ops M) c (inputs c (memory s)) scale (s.pids 2) n
    (s.pids 1*c.BK+i.2.1.val) (s.pids 0*c.BV+i.1.val)

def rowValues (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat) :
    Values α .real [c.BV] :=
  fun i => (ops M).storeCast
    (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (terms M c s scale n) i)

/-- Only the indicated source row is written. The values come from the
independent input-array recurrence above. -/
def rowWrite (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat)
    (m : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [c.BV]).foldl (fun current i =>
    if valueMask c s i.1 then
      fun r b => if r="o" ∧ b=outputBase c s i.1+visit c n*c.V then
        .mk .real (rowValues M c s scale n i) else current r b
    else current) m

def loopMemory (M : Algebra α) (c : Shape) (s : State α) (scale : α) :
    Nat → RegionName → Nat → Cell α
  | 0 => s.mem
  | n+1 => rowWrite M c s scale n (loopMemory M c s scale n)

theorem rowWrite_input (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat)
    (m : RegionName → Nat → Cell α) (r : RegionName) (hr : r≠"o") (b : Nat) :
    rowWrite M c s scale n m r b=m r b := by
  unfold rowWrite
  generalize TileShape.allIndices [c.BV] = lanes
  induction lanes generalizing m with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split <;> simp [hr]

theorem loopMemory_input (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (n : Nat) (r : RegionName) (hr : r≠"o") (b : Nat) :
    loopMemory M c s scale n r b=s.mem r b := by
  induction n with
  | zero => rfl
  | succ n ih => rw [loopMemory,rowWrite_input _ _ _ _ _ _ _ hr,ih]

structure Context (M : Algebra α) (c : Shape) (initial : State α) (scale : α)
    (n : Nat) (current : State α) : Prop where
  mem : current.mem=loopMemory M c initial scale n
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 0)
  ik : current.regs .nat [] "i_k"=some (fun _ => initial.pids 1)
  iz : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  km : current.regs .bool [c.BK] "mask_bk"=some (fun i => keyMask c initial i.1)
  vm : current.regs .bool [c.BV] "mask_bv"=some (fun i => valueMask c initial i.1)
  kvm : current.regs .bool [c.BV,c.BK] "mask_kv"=
    some (fun i => valueMask c initial i.1 && keyMask c initial i.2.1)
  pq : current.regs .ptr [c.BK] "p_q"=
    some (fun i : TileIndex [c.BK] => (("q" : RegionName),pointer c (keyBase c initial i.1) c.K n))
  pk : current.regs .ptr [c.BK] "p_k"=
    some (fun i : TileIndex [c.BK] => (("k" : RegionName),pointer c (keyBase c initial i.1) c.K n))
  pw : current.regs .ptr [c.BK] "p_w"=
    some (fun i : TileIndex [c.BK] => (("w" : RegionName),pointer c (keyBase c initial i.1) c.K n))
  pv : current.regs .ptr [c.BV] "p_v"=
    some (fun i : TileIndex [c.BV] => (("v" : RegionName),pointer c (valueBase c initial i.1) c.V n))
  po : current.regs .ptr [c.BV] "p_o"=
    some (fun i : TileIndex [c.BV] => (("o" : RegionName),pointer c (outputBase c initial i.1) c.V n))
  bias : current.regs .real [c.BK] "b_u"=some (Source.bias M c initial)
  state : current.regs .real [c.BV,c.BK] "b_h"=some (carry M c initial n)
  scalar : current.regs .real [] "scale"=some (fun _ => scale)

theorem entry_context (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (hs : s.regs .real [] "scale"=some (fun _ => scale)) :
    Context M c s scale 0 (entry M c s).val := by
  by_cases hi : c.initial=Bool.true
  all_goals by_cases hr : c.reverse=Bool.true
  all_goals constructor
  all_goals simp [entry,hi,hr,carry,scalarState,scalarSeed,ops,memory,regionName,
    inputs,stateAddress,biasAddress,loopMemory,bias,keyMask,valueMask,
    pointer,keyBase,valueBase,outputBase,hs,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,show (0.0 : ℝ)=0 by norm_num]
  all_goals try funext i
  all_goals try simp [Source.carry,scalarState,scalarSeed,ops,memory,regionName,
    inputs,stateAddress,biasAddress,Source.bias,keyMask,valueMask,hi,hr,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,
    and_comm,add_ite,ite_add]
  all_goals simp only [sub_mul,one_mul,add_sub_assoc,
    ←Int.natCast_mul,←Int.natCast_add,←Int.natCast_emod,
    Int.toNat_natCast,Int.toNat_sub,Int.ofNat_lt]
  all_goals simp [Nat.add_mul,Nat.mul_add,Nat.mul_assoc,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals simp only [←add_sub_assoc,←Int.natCast_mul,←Int.natCast_add,Int.toNat_sub]
  all_goals omega

#specNonCircular terms avoiding [run,FP.Structural.exec,entry,loopMemory]
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
