import bench.repairs.tritonbench_g.cosine_compute.Proofs.LegacySource
import bench.repairs.tritonbench_g.cosine_compute.Proofs.Index
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Scheduling
import bench.tritonbench_g.cosine_compute.Proofs.NativeSource
import VeriTile.Torch.FP32

/-! Specialize the existing source parser and scalar semantics to the unchanged
FP32 kernel. Native signed-i32 offsets and sign-extended byte pointers are
interpreted before reusing the generic unary store/schedule proof. -/
namespace VeriTile.Bench.Repairs.CosineCompute.Native
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

abbrev Word := VeriTile.Torch.FP32.Word
abbrev Memory := TritonMul2.Memory Word
abbrev Layout := TritonMul2.Layout
abbrev Allocated := TritonMul2.Allocated

def scalar (b : CosineNative.Numerics) (x : Word) : Word :=
  ⟨(b.cos32 ⟨x.bits⟩).bits⟩
def operations (b : CosineNative.Numerics) : TritonMul2.Operations Word := ⟨scalar b⟩
def code (n : Nat) := CosineNative.sourceCode n 256

theorem source_lowering (n : Nat) :
    CosineNative.lowerBody (CosineNative.sourceStatements n 256) = some (code n) :=
  CosineNative.original_syntax_lowering n 256

def scalarExpr (n : Nat) : CosineNative.Expr .fp32 (.storage .fp32) :=
  (((code n)[3]?).bind (CosineNative.lowerScalar .fp32)).get (by
    simp [code,CosineNative.sourceCode,CosineNative.lowerScalar])

theorem scalar_source (b : CosineNative.Numerics) (n : Nat) (x : Word) :
    (CosineNative.Expr.eval (d:=.fp32) b (⟨x.bits⟩ : CosineNative.Word (.storage .fp32)) (scalarExpr n)).map (fun v => (⟨v.bits⟩ : Word)) = some (scalar b x) := by
  rfl

def registers (n pid : Nat) := CosineNative.runPrefix pid (code n)
def pointer (base n pid lane : Nat) : Nat :=
  (BitVec.ofNat 64 (4*base) + ((registers n pid).indices "offset" lane).signExtend 64 * 4).toNat / 4

theorem source_indices (n pid lane : Nat) :
    (registers n pid).indices "offset" lane = offset32 pid lane ∧
    (registers n pid).masks "mask" lane = mask32 n pid lane := by
  simp only [registers,code,CosineNative.sourceCode,CosineNative.runPrefix,List.foldl_cons,
    List.foldl_nil,CosineNative.stepPrefix,CosineNative.evalIndex,CosineNative.evalMask]
  simp only [if_true]
  exact ⟨rfl,rfl⟩

theorem pointer_exact (base n pid lane : Nat) (hn : n < 2^31)
    (hp : pid < gridCount n) (hl : lane < 256) (active : pid*256+lane < n)
    (span : base+n ≤ 2^62) : pointer base n pid lane = base+pid*256+lane := by
  have idx := indices n pid lane hn hp hl
  have ext : (offset32 pid lane).signExtend 64 = BitVec.ofNat 64 (pid*256+lane) := by
    apply BitVec.eq_of_toInt_eq
    rw [BitVec.toInt_signExtend_of_le (by decide),idx.2.1]
    exact (BitVec.toInt_ofInt_eq_self (w:=64) (n:=((pid*256+lane : Nat) : Int))
      (by decide) (by omega) (by norm_num; omega)).symm
  have bound : 4*base+(pid*256+lane)*4 < 2^64 := by omega
  unfold pointer
  rw [(source_indices n pid lane).1,ext]
  change (BitVec.ofNat 64 (4*base) + BitVec.ofNat 64 (pid*256+lane) * BitVec.ofNat 64 4).toNat/4 = _
  rw [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]
  omega

/-- Executes the source index/mask registers, masked load, explicit FP32 cast and
tl.cos, then masked FP32 stores. The cast and destination conversion are identity
on FP32 words; inactive unspecified values have no store observation. -/
def program (b : CosineNative.Numerics) (l : Layout) (pid : Nat) (s : Memory) : Memory :=
  TritonMul2.stores ((List.range 256).filterMap fun lane =>
    if (registers l.count pid).masks "mask" lane then
      (CosineNative.Expr.eval (d:=.fp32) b (⟨(s (pointer l.input l.count pid lane)).bits⟩ : CosineNative.Word (.storage .fp32)) (scalarExpr l.count)).map
        (fun v => (pointer l.output l.count pid lane,(⟨v.bits⟩ : Word)))
    else none) s

theorem program_eq (b : CosineNative.Numerics) (l : Layout)
    (allocation : Allocated l) (block : l.block=256) (hn : l.count < 2^31)
    (pid : Nat) (hp : pid < gridCount l.count) (s : Memory) :
    program b l pid s = TritonMul2.program (operations b) l pid s := by
  unfold program TritonMul2.program
  rw [block]
  congr 1
  apply List.filterMap_congr
  intro lane member
  have hl : lane < 256 := List.mem_range.mp member
  rw [(source_indices l.count pid lane).2,(indices l.count pid lane hn hp hl).2.2]
  by_cases active : pid*256+lane < l.count
  · simp only [active,decide_true,↓reduceIte]
    rw [pointer_exact l.input l.count pid lane hn hp hl active allocation.input,
      pointer_exact l.output l.count pid lane hn hp hl active allocation.output]
    simp only [Nat.add_assoc]
    rfl
  · simp [active]

def execute (b : CosineNative.Numerics) (l : Layout) (order : List Nat) (s : Memory) : Memory :=
  order.foldl (fun memory pid => program b l pid memory) s

theorem execute_eq (b : CosineNative.Numerics) (l : Layout)
    (allocation : Allocated l) (block : l.block=256) (hn : l.count < 2^31)
    (order : List Nat) (valid : ∀ p∈order, p < gridCount l.count) (s : Memory) :
    execute b l order s = TritonMul2.Scheduling.execute (operations b) l order s := by
  induction order generalizing s with
  | nil => rfl
  | cons p rest ih =>
    change execute b l rest (program b l p s) =
      TritonMul2.Scheduling.execute (operations b) l rest (TritonMul2.program (operations b) l p s)
    rw [program_eq b l allocation block hn p (valid p (by simp))]
    exact ih (fun q hq=>valid q (by simp [hq])) _

theorem launch_correct (b : CosineNative.Numerics) (l : Layout)
    (allocation : Allocated l) (block : l.block=256) (hn : l.count < 2^31)
    (order : List Nat) (complete : order.Perm (List.range (gridCount l.count))) (s : Memory) :
    (∀ i, i < l.count → execute b l order s (l.output+i)=scalar b (s (l.input+i))) ∧
    (∀ cell, (¬∃ i, i < l.count ∧ cell=l.output+i) → execute b l order s cell=s cell) := by
  rw [execute_eq b l allocation block hn order (fun p hp=>
    List.mem_range.mp (complete.mem_iff.mp hp))]
  have grid_eq : TritonMul2.grid l.count l.block = gridCount l.count := by
    simp [TritonMul2.grid,gridCount,block]
  rw [TritonMul2.Scheduling.order_equal (operations b) l allocation order (by rwa [grid_eq])]
  exact TritonMul2.launch_correct (operations b) l allocation (by omega) s

#axiomsClean source_lowering
#axiomsClean scalar_source
#axiomsClean pointer_exact
#axiomsClean program_eq
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.Native
