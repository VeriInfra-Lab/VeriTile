import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Contract
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Host

/-! The i32 program and lane registers are sign-extended before i64 arithmetic.
The finite interpreter lowers and executes the actual complete typed source;
only the ordered scalar multiply is supplied as an operation. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Native
open VeriTile.Triton
open VeriTile.Bench.Repairs.TritonMul2 (Operations Layout Allocated Memory stores program launch launch_correct)
open TritonMul2Finite (IndexExpr MaskExpr lowerIndex lowerBool lowerMask)
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
local instance : DecidableEq ℝ := Classical.decEq _

def offset (B pid lane : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 pid).signExtend 64 * BitVec.ofNat 64 B + (BitVec.ofNat 32 lane).signExtend 64

def active (n B pid lane : Nat) : Bool := decide ((offset B pid lane).toInt < (BitVec.ofNat 64 n).toInt)

theorem nat64_signed (n : Nat) (h : n<2^63) : (BitVec.ofNat 64 n).toInt=(n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)

theorem widen32 (n : Nat) (h : n<2^31) :
    (BitVec.ofNat 32 n).signExtend 64=BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),TritonMul2.nat32_signed n h,nat64_signed n (by omega)]

theorem indices (n B pid lane : Nat) (extent : n≤2^62) (blocks : B∈supportedBlocks)
    (gridBound : grid n B≤2^31-1) (hp : pid<grid n B) (hl : lane<B) :
    (offset B pid lane).toInt=(pid*B+lane : Nat) ∧ active n B pid lane=decide (pid*B+lane<n) := by
  have bb:=block_bound B blocks
  have pb : pid<2^31 := by omega
  have lb : lane<2^31 := by omega
  have ob : pid*B+lane<2^51 := by
    calc pid*B+lane < (pid+1)*B := by rw [Nat.add_mul];simp only [Nat.one_mul];omega
         _ ≤ 2^31*2^20 := Nat.mul_le_mul (by omega) bb
         _ = 2^51 := by norm_num
  have eq : (offset B pid lane).toInt=(pid*B+lane : Nat) := by
    rw [offset,widen32 pid pb,widen32 lane lb,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    exact nat64_signed _ (by omega)
  exact ⟨eq,by simp only [active,eq,nat64_signed n (by omega),Int.ofNat_lt]⟩

def address (base : Nat) (index : BitVec 64) : Nat :=
  (BitVec.ofInt 64 (4*(base : Int)+4*index.toInt)).toNat/4

theorem address_exact (base n i : Nat) (allocation : base+n≤2^62) (inside : i<n)
    (index : BitVec 64) (eq : index.toInt=(i : Int)) : address base index=base+i := by
  unfold address
  rw [eq]
  simp only [BitVec.toNat_ofInt,Nat.cast_pow,Nat.cast_ofNat]
  rw [Int.emod_eq_of_lt (by omega : 0≤4*(base : Int)+4*(i : Int))
    (by omega : 4*(base : Int)+4*(i : Int)<2^64)]
  have cast : 4*(base : Int)+4*(i : Int)=((4*(base+i) : Nat) : Int) := by push_cast;ring
  rw [cast,Int.toNat_natCast]
  omega

inductive Instruction where
  | constant (name : RegName)
  | indexAssign (name : RegName) (value : IndexExpr)
  | maskAssign (name : RegName) (value : MaskExpr)
  | load (name : RegName) (region : RegionName) (offset : IndexExpr) (mask : MaskExpr)
  | multiply (name input constant : RegName)
  | store (region : RegionName) (offset : IndexExpr) (name : RegName) (mask : MaskExpr)

def lowerStatement : Stmt → Option Instruction
  | .assign .real _ name (.const v) => if v=exponentCompensator then some (.constant name) else none
  | .assign .nat _ name e => return .indexAssign name (←lowerIndex e)
  | .assign .bool _ name e => return .maskAssign name (←lowerBool e)
  | .assign .real _ name (.load .real (.region region offset) mask _) =>
      return .load name region (←lowerIndex offset) (←lowerMask mask)
  | .assign .real _ name (.mul .real _ (.ref _ _ input) (.ref _ _ constant)) =>
      some (.multiply name input constant)
  | .store .real _ (.region region offset) (.ref _ _ name) mask =>
      return .store region (←lowerIndex offset) name (←lowerMask mask)
  | _ => none

def lowerBody : List Stmt → Option (List Instruction)
  | [] => some []
  | stmt::rest => return (←lowerStatement stmt)::(←lowerBody rest)

def sourceCode (n B : Nat) : List Instruction :=
  [.constant "exponent_compensator",
   .indexAssign "idxs" (.add (.mul (.program 0) (.literal B)) (.lane B)),
   .maskAssign "mask" (.less (.register "idxs") (.literal n)),
   .load "x" "src" (.register "idxs") (.register "mask"),
   .multiply "y" "x" "exponent_compensator",
   .store "dst" (.register "idxs") "y" (.register "mask")]

theorem source_lowering (n B : Nat) :
    lowerBody (candidateKernel "src" "dst" n B).body=some (sourceCode n B) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerMask,lowerBool,candidateKernel,
    ComputeKernel.body,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,
    bind,pure,Except.bind,Option.bind,ComputeKernel.eraseDType,ComputeDType.eraseDType,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?]

def evalIndex (pid lane : Nat) (regs : RegName → Nat → BitVec 64) : IndexExpr → BitVec 64
  | .literal n => BitVec.ofNat 64 n
  | .program ax => if ax=0 then (BitVec.ofNat 32 pid).signExtend 64 else 0
  | .register name => regs name lane
  | .lane _ => (BitVec.ofNat 32 lane).signExtend 64
  | .add a b => evalIndex pid lane regs a + evalIndex pid lane regs b
  | .mul a b => evalIndex pid lane regs a * evalIndex pid lane regs b

def evalMask (pid lane : Nat) (indices : RegName → Nat → BitVec 64)
    (masks : RegName → Nat → Bool) : MaskExpr → Bool
  | .register name => masks name lane
  | .less a b => decide ((evalIndex pid lane indices a).toInt < (evalIndex pid lane indices b).toInt)

def nativeProgram (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α) : Memory α :=
  stores ((List.range l.block).filterMap fun lane =>
    if active l.count l.block pid lane then
      some (address l.output (offset l.block pid lane),ops.twice (s (address l.input (offset l.block pid lane))))
    else none) s

def executeCode (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α)
    (constants : RegName → Bool) (indices : RegName → Nat → BitVec 64)
    (masks : RegName → Nat → Bool) : List Instruction → Option (Memory α)
  | [] => some s
  | .constant name :: rest => executeCode ops l pid s
      (fun key => if key=name then Bool.true else constants key) indices masks rest
  | .indexAssign name e :: rest => executeCode ops l pid s constants
      (fun key lane => if key=name then evalIndex pid lane indices e else indices key lane) masks rest
  | .maskAssign name e :: rest => executeCode ops l pid s constants indices
      (fun key lane => if key=name then evalMask pid lane indices masks e else masks key lane) rest
  | [.load name region inputOffset mask,.multiply value operand constant,.store target outputOffset storedName outputMask] =>
      if region="src" ∧ target="dst" ∧ name=operand ∧ value=storedName ∧ constants constant ∧ mask=outputMask then
        some (stores ((List.range l.block).filterMap fun lane =>
          if evalMask pid lane indices masks mask then
            some (address l.output (evalIndex pid lane indices outputOffset),
              ops.twice (s (address l.input (evalIndex pid lane indices inputOffset))))
          else none) s)
      else none
  | _ => none

def sourceProgram (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α) : Option (Memory α) := do
  let code ← lowerBody (candidateKernel "src" "dst" l.count l.block).body
  executeCode ops l pid s (fun _ => Bool.false) (fun _ _ => 0) (fun _ _ => Bool.false) code

theorem source_reduction (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α) :
    sourceProgram ops l pid s=some (nativeProgram ops l pid s) := by
  simp only [sourceProgram,source_lowering,bind,Option.bind,sourceCode,executeCode,
    evalIndex,evalMask,String.reduceEq,if_true,if_false,and_self,nativeProgram,active,offset]
  rfl

theorem program_equal (ops : Operations α) (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block≤2^31-1)
    (allocation : Allocated l) (pid : Nat) (hp : pid<grid l.count l.block) (s : Memory α) :
    nativeProgram ops l pid s=program ops l pid s := by
  unfold nativeProgram program
  congr 1
  apply List.filterMap_congr
  intro lane hl
  have extent : l.count≤2^62 := by have := allocation.input;omega
  obtain ⟨index,mask⟩ := indices l.count l.block pid lane extent blocks gridBound hp (List.mem_range.mp hl)
  rw [mask]
  by_cases act : pid*l.block+lane<l.count
  · simp only [act,decide_true,if_true]
    rw [address_exact _ _ _ allocation.output act _ index,address_exact _ _ _ allocation.input act _ index]
  · simp [act]

def sourceLaunch (ops : Operations α) (l : Layout) : List Nat → Memory α → Option (Memory α)
  | [], s => some s
  | pid::rest, s => do
      let next ← sourceProgram ops l pid s
      sourceLaunch ops l rest next

theorem source_launch_equal (ops : Operations α) (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block≤2^31-1) (allocation : Allocated l)
    (order : List Nat) (valid : ∀ p∈order,p<grid l.count l.block) (s : Memory α) :
    sourceLaunch ops l order s=some (TritonMul2.Scheduling.execute ops l order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
    rw [sourceLaunch,source_reduction,program_equal ops l blocks gridBound allocation p (valid p (by simp))]
    exact ih (fun q hq => valid q (by simp [hq])) _

theorem source_launch_correct (ops : Operations α) (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block≤2^31-1) (allocation : Allocated l)
    (order : List Nat) (schedule : order.Perm (List.range (grid l.count l.block))) (s : Memory α) :
    sourceLaunch ops l order s=some (launch ops l s) ∧
    (∀ i,i<l.count → launch ops l s (l.output+i)=ops.twice (s (l.input+i))) ∧
    (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → launch ops l s cell=s cell) := by
  refine ⟨?_,launch_correct ops l allocation (block_positive _ blocks) s⟩
  rw [source_launch_equal ops l blocks gridBound allocation order
    (fun p hp => List.mem_range.mp (schedule.mem_iff.mp hp)),TritonMul2.Scheduling.order_equal ops l allocation order schedule]

#axiomsClean indices
#axiomsClean address_exact
#axiomsClean source_lowering
#axiomsClean source_launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Native
