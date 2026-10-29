import bench.repairs.tritonbench_g.triton_mul2.Proofs.Contract
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Scheduling
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Projection

/-! Finite execution of the unchanged lowered AST. Index registers are i32,
comparison is signed, and element pointers use signed extension into u64 bytes.
Only the scalar FP32 multiplication is abstracted by Operations.twice. -/
namespace VeriTile.Bench.Repairs.TritonMul2.Native
open VeriTile.Triton
open TritonMul2Finite (IndexExpr MaskExpr Instruction evalIndex evalMask)
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def address (base : Nat) (index : BitVec 32) : Nat :=
  (BitVec.ofInt 64 (4*(base : Int)+4*index.toInt)).toNat/4

theorem address_exact (base n i : Nat) (allocation : base+n≤2^62) (inside : i<n)
    (index : BitVec 32) (eq : index.toInt=(i : Int)) : address base index=base+i := by
  unfold address
  rw [eq]
  simp only [BitVec.toNat_ofInt,Nat.cast_pow,Nat.cast_ofNat]
  rw [Int.emod_eq_of_lt (by omega : 0≤4*(base : Int)+4*(i : Int))
    (by omega : 4*(base : Int)+4*(i : Int)<2^64)]
  have cast : 4*(base : Int)+4*(i : Int)=((4*(base+i) : Nat) : Int) := by push_cast;ring
  rw [cast,Int.toNat_natCast]
  omega

def nativeProgram (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α) : Memory α :=
  stores ((List.range l.block).filterMap fun lane =>
    if TritonMul2Finite.active32 l.count l.block pid lane then
      some (address l.output (TritonMul2Finite.offset32 l.block pid lane),
        ops.twice (s (address l.input (TritonMul2Finite.offset32 l.block pid lane))))
    else none) s

/-- Ordinary register evaluation is reused from the original finite source.
The final three instructions take one whole load snapshot before any writes. -/
def executeCode (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α)
    (outputName : RegionName) (indices : RegName → Nat → BitVec 32)
    (masks : RegName → Nat → Bool) : List Instruction → Option (Memory α)
  | [] => some s
  | .indexAssign name e :: rest => executeCode ops l pid s outputName
      (fun key lane => if key=name then evalIndex pid lane indices e else indices key lane) masks rest
  | .maskAssign name e :: rest => executeCode ops l pid s outputName indices
      (fun key lane => if key=name then evalMask pid lane indices masks e else masks key lane) rest
  | [.load name region offset mask,.double value operand,.store target outputOffset storedName outputMask] =>
      if region="input" ∧ target=outputName ∧ name=operand ∧ value=storedName ∧ mask=outputMask then
        some (stores ((List.range l.block).filterMap fun lane =>
          if evalMask pid lane indices masks mask then
            some (address l.output (evalIndex pid lane indices outputOffset),
              ops.twice (s (address l.input (evalIndex pid lane indices offset))))
          else none) s)
      else none
  | _ => none

def sourceProgram (ops : Operations α) (l : Layout) (inplace : Bool) (pid : Nat)
    (s : Memory α) : Option (Memory α) := do
  let code ← TritonMul2Finite.lowerBody (if inplace then
    (inplaceSourceKernel "input" l.count l.block).body else
    (sourceKernel "input" "output" l.count l.block).body)
  executeCode ops l pid s (if inplace then "input" else "output")
    (fun _ _ => 0) (fun _ _ => Bool.false) code

theorem source_reduction (ops : Operations α) (l : Layout) (inplace : Bool) (pid : Nat)
    (s : Memory α) : sourceProgram ops l inplace pid s=some (nativeProgram ops l pid s) := by
  cases inplace <;>
    simp only [sourceProgram, Bool.false_eq_true, if_false, if_true,
      Projection.original_body, Projection.inplace_body,
      TritonMul2Finite.original_ast_lowering,TritonMul2Finite.inplace_ast_lowering,
      bind,Option.bind,TritonMul2Finite.sourceCode,executeCode,evalIndex,evalMask,
      String.reduceEq,if_true,if_false,and_self,nativeProgram,
      TritonMul2Finite.active32,TritonMul2Finite.offset32] <;> rfl

theorem program_equal (ops : Operations α) (l : Layout)
    (extent : l.count<2^31) (blocks : l.block∈supportedBlocks) (allocation : Allocated l)
    (pid : Nat) (hp : pid<grid l.count l.block) (s : Memory α) :
    nativeProgram ops l pid s=program ops l pid s := by
  unfold nativeProgram program
  congr 1
  apply List.filterMap_congr
  intro lane hl
  obtain ⟨index,mask⟩ := native_indices l.count l.block pid lane extent blocks hp (List.mem_range.mp hl)
  rw [mask]
  by_cases active : pid*l.block+lane<l.count
  · simp only [active,decide_true,if_true]
    rw [address_exact _ _ _ allocation.output active _ index,
      address_exact _ _ _ allocation.input active _ index]
  · simp [active]

def sourceLaunch (ops : Operations α) (l : Layout) (inplace : Bool) : List Nat → Memory α → Option (Memory α)
  | [], s => some s
  | pid::rest, s => do
      let next ← sourceProgram ops l inplace pid s
      sourceLaunch ops l inplace rest next

theorem source_launch_equal (ops : Operations α) (l : Layout) (inplace : Bool)
    (extent : l.count<2^31) (blocks : l.block∈supportedBlocks) (allocation : Allocated l)
    (order : List Nat) (valid : ∀ p∈order,p<grid l.count l.block) (s : Memory α) :
    sourceLaunch ops l inplace order s=some (Scheduling.execute ops l order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
    rw [sourceLaunch,source_reduction,program_equal ops l extent blocks allocation p (valid p (by simp))]
    exact ih (fun q hq => valid q (by simp [hq])) _

theorem source_launch_correct (ops : Operations α) (l : Layout) (inplace : Bool)
    (extent : l.count<2^31) (blocks : l.block∈supportedBlocks) (allocation : Allocated l)
    (order : List Nat) (schedule : order.Perm (List.range (grid l.count l.block))) (s : Memory α) :
    sourceLaunch ops l inplace order s=some (launch ops l s) ∧
    Doubled ops.twice l.count (fun i => s (l.input+i)) (fun i => launch ops l s (l.output+i)) ∧
    (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → launch ops l s cell=s cell) := by
  refine ⟨?_,launch_correct ops l allocation (block_positive _ blocks) s⟩
  rw [source_launch_equal ops l inplace extent blocks allocation order
    (fun p hp => List.mem_range.mp (schedule.mem_iff.mp hp)),Scheduling.order_equal ops l allocation order schedule]

/-- Same-pointer wiring is a host obligation for the in-place source. -/
def Wired (inplace : Bool) (l : Layout) : Prop := inplace=Bool.true → l.input=l.output

#axiomsClean address_exact
#axiomsClean source_reduction
#axiomsClean source_launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonMul2.Native
