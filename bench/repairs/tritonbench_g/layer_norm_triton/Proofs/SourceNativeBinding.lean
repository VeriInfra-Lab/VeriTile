import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeStorage
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Machine
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceRows

/-! Local compiler and ABI bindings. The decoder returns only structural
policy, addresses and control; it cannot return an execution or output word.
All numerical target behavior is carried by individual instruction primitives. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeBinding
open VeriTile.Triton
open SourceNative.Native SourceNativeStorage SourceStrideRefinement
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section

structure Lowering where
  entry : Compiler.Entry
  addresses : List Compiler.Address
  control : Compiler.Control
  integerWidth : Nat
  fp32Loads : Bool
  fp32Carries : Bool
  fp32Stores : Bool
  reductions : Reductions
  schedule : OutputSchedule

structure Platform where
  decode : ByteArray → Compiler.Signature → Compiler.Options → Compiler.Primitives → Option Lowering
  instructions : Compiler.Primitives → Instructions
  epsilon : Compiler.Object → UInt64 → Word
  epsilonBinding : Compiler.Shared.Binding → UInt64 → Prop
  defaults : Compiler.Entry → Nat → Compiler.Options
  warps : ∀ e B,(defaults e B).numWarps=(if e=.finalBackward then 4 else Geometry.warps B)

structure Selected (platform : Platform) (entry : Compiler.Entry) (p : Policy) where
  object : Compiler.Object
  source : object.sourceHash=Source.candidateSHA256
  entryEq : object.entry=entry
  lowering : Lowering
  decoded : platform.decode object.bytes object.signature object.options object.primitives=some lowering
  decodedEntry : lowering.entry=entry
  addresses : lowering.addresses=Compiler.addresses entry
  control : lowering.control=(match entry with | .forward => .oneRow | .groupedBackward => .groupedRows | .finalBackward => .tiledGroups)
  precision : lowering.integerWidth=64 ∧ lowering.fp32Loads=Bool.true ∧ lowering.fp32Carries=Bool.true ∧ lowering.fp32Stores=Bool.true
  primitives : p.instructions=platform.instructions object.primitives
  reductions : p.reductions=lowering.reductions
  schedule : p.schedule=lowering.schedule

structure Bound (platform : Platform) (code : Selected platform entry p) (stride M N B : Nat) (eps : Float) : Prop where
  scalarArguments : List.Forall₂ Compiler.Shared.integerBinding code.object.signature.scalars
    (match entry with | .forward => [stride,N] | .groupedBackward => [stride,M,N] | .finalBackward => [min (Geometry.groups N) M,N])
  pointerTypes : code.object.signature.pointerBits=List.replicate (if entry=.groupedBackward then 8 else if entry=.forward then 6 else 4) 32
  block : code.object.signature.block=(if entry=.finalBackward then 128 else B)
  groups : entry=.groupedBackward → code.object.signature.group=Geometry.groups N
  rowBlock : entry=.finalBackward → code.object.signature.rowBlock=32
  epsilonABI : entry=.forward → ∃ b,code.object.signature.epsilon=some b ∧ platform.epsilonBinding b eps.toBits
  options : code.object.options=platform.defaults entry B

/-- Exact host binary64 bits, including admitted values that convert to zero.
This is one literal-conversion law and has no output or execution premise. -/
def Epsilon (platform : Platform) (code : Selected platform .forward p) (value : ℝ) (host : Float) : Prop :=
  VeriTile.Bench.Repairs.FastLayernorm.KernelNative.literal32 p.instructions value=platform.epsilon code.object host.toBits

theorem epsilon_operand (platform : Platform) (code : Selected platform .forward p) (value : ℝ) (host : Float)
    (h : Epsilon platform code value host) (s : Schedule) :
    eval p.instructions s (.literal (some .fp32) .real value)=platform.epsilon code.object host.toBits := by
  rw [VeriTile.Bench.Repairs.FastLayernorm.KernelNative.eval_literal32]
  exact h

theorem signed64 (n : Nat) (hn : n<2^63) : (BitVec.ofNat 64 n).toInt=(n:Int) := by
  have hm : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,hm]
  rw [hm]
  norm_num at *
  omega

theorem signed_lt (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (BitVec.ofNat 64 a).slt (BitVec.ofNat 64 b)=decide (a<b) := by
  rw [BitVec.slt_eq_decide,signed64 a ha,signed64 b hb]
  simp

/-- The final failed while test and every entered row increment retain the
natural positive-step loop meaning in signed64 arithmetic. -/
theorem grouped_loop (x : KernelMemory.Input) (M N row : Nat) (shape : x.dense.count=M*N)
    (hn : 0<N) (hr : row≤M+Geometry.groups N) :
    (BitVec.ofNat 64 row).slt (BitVec.ofNat 64 M)=decide (row<M) ∧
      (row<M → (BitVec.ofNat 64 row+BitVec.ofNat 64 (Geometry.groups N)).toInt=(row+Geometry.groups N:Int)) := by
  have count := dense_count_bound x.dense
  have size : M≤M*N := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left M (Nat.succ_le_of_lt hn)
  rw [shape] at count
  have groups := (Geometry.groups_bounds N).2
  have hm : M<2^63 := by norm_num at *; omega
  have hrow : row<2^63 := by norm_num at *; omega
  refine ⟨signed_lt row M hrow hm,?_⟩
  intro h
  rw [←BitVec.ofNat_add]
  apply signed64
  norm_num at *
  omega

/-- Actual observed strides, including arbitrary singleton strides, have the
same signed64 address effect as the dense logical coordinate. -/
theorem row_address (c : KernelContract.Forward M N) (view : MatrixView c.x M N)
    (B W : Nat) (settings : Compiler.settings N=.ok (B,W)) (hn : 0<N)
    (row col : Nat) (hr : row<M) (hc : col<B) :
    (BitVec.ofNat 64 row*BitVec.ofNat 64 view.stride+BitVec.ofNat 64 col).toInt=(row*N+col:Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,view.row_effect hn row hr]
  exact signed64 _ (Geometry.raw_row_offset_safe c.x.dense M N row col B c.xCount hr hc
    (Compiler.settings_success N B W settings).2.2.1)

theorem backward_address (x w dy r mu : KernelMemory.Input)
    (c : KernelContract.Backward M N x w dy r mu) (view : MatrixView x M N)
    (B W : Nat) (settings : Compiler.settings N=.ok (B,W)) (hn : 0<N)
    (row col : Nat) (hr : row<M) (hc : col<B) :
    (BitVec.ofNat 64 row*BitVec.ofNat 64 view.stride+BitVec.ofNat 64 col).toInt=(row*N+col:Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,view.row_effect hn row hr]
  exact signed64 _ (Geometry.raw_row_offset_safe x.dense M N row col B c.xCount hr hc
    (Compiler.settings_success N B W settings).2.2.1)

theorem partial_address (x w dy r mu : KernelMemory.Input)
    (c : KernelContract.Backward M N x w dy r mu)
    (B W : Nat) (settings : Compiler.settings N=.ok (B,W))
    (group col : Nat) (hg : group<min (Geometry.groups N) M) (hc : col<B) :
    (BitVec.ofNat 64 group*BitVec.ofNat 64 N+BitVec.ofNat 64 col).toInt=(group*N+col:Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact signed64 _ (Geometry.raw_row_offset_safe (c.first.buffers.output 1)
    (min (Geometry.groups N) M) N group col B c.first.dwCount hg hc
    (Compiler.settings_success N B W settings).2.2.1)

theorem final_address (N pid iteration laneRow laneCol : Nat) (hn : N≤16384)
    (hp : pid<(N+127)/128) (hi : iteration<(min (Geometry.groups N) M+31)/32)
    (hr : laneRow<32) (hc : laneCol<128) :
    (BitVec.ofNat 64 (iteration*32+laneRow)*BitVec.ofNat 64 N+
      BitVec.ofNat 64 (pid*128+laneCol)).toInt=((iteration*32+laneRow)*N+(pid*128+laneCol):Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact signed64 _ (Geometry.final_raw_offset_safe N _ _ hn
    (Geometry.group_tile_counter_safe N M iteration laneRow hi hr) (Geometry.final_column_safe N pid laneCol hn hp hc))

theorem grouped_rows (M N group : Nat) :
    SourceRows.rows M (Geometry.groups N) group=groupRows M (Geometry.groups N) group :=
  SourceRows.rows_eq M (Geometry.groups N) group (Geometry.groups_bounds N).1

#axiomsClean epsilon_operand
#axiomsClean signed_lt
#axiomsClean grouped_loop
#axiomsClean row_address
#axiomsClean backward_address
#axiomsClean partial_address
#axiomsClean final_address
#axiomsClean grouped_rows
#specNonCircular Epsilon avoiding [FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNativeBinding
