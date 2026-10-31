import bench.tritonbench_g.relu_triton_kernel.Kernels
import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_GatedGrid

/-! Typed source materializations for the twelve admitted logical profiles.
Integer width names retain their source identity but use decoded Int/Nat
cells; native width/storage decoding is a separate obligation. No fp64 or
fp8 execution is silently represented by the real channel. Integer-to-fp32
conversion uses fromInt/fromNat, with an explicit identity carrier retag.
The bf16 comparison widens to fp32 while selection remains bf16.
-/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
open VeriTile.Triton FP.Structural GatedGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Kind where
  | signed | unsigned | boolean | fp16 | bf16 | fp32
  deriving DecidableEq, Repr

inductive Profile where
  | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | boolean | fp16 | bf16 | fp32
  deriving DecidableEq, Repr

def Profile.kind : Profile → Kind
  | .i8 | .i16 | .i32 | .i64 => .signed
  | .u8 | .u16 | .u32 | .u64 => .unsigned
  | .boolean => .boolean | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32

def Kind.inputType : Kind → TileDType
  | .signed => .int | .unsigned => .nat | .boolean => .bool
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32

def Kind.resultType : Kind → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | _ => .fp32

def backend {α : Type} (M : Algebra α) (cmp16 cmp32 : α → α → Bool) : Algebra α :=
  { M with
    compareLe := fun p d =>
      if d = .fp16 then some cmp16 else if d = .fp32 then some cmp32 else M.compareLe p d
    cast := fun p src dst a =>
      if src = dst ∨ (src = .real ∧ dst = .fp32) then a else M.cast p src dst a }

def boolNat (B : Nat) : Op .nat [B] :=
  .where (.ref .bool [B] "x") (.full [B] (.constNat 1)) (.full [B] (.constNat 0))

def resultOp (k : Kind) (B : Nat) : Op k.resultType [B] :=
  match k with
  | .signed => .where
      (.ge .int .scalarR (.ref .int [B] "x") (.constInt 0))
      (.castFloat .real .fp32 (.intToReal (.ref .int [B] "x")))
      (.broadcast (.constFloat .fp32 0) [B])
  | .unsigned => .where
      (.ge .nat .scalarR (.ref .nat [B] "x") (.constNat 0))
      (.castFloat .real .fp32 (.natToReal (.ref .nat [B] "x")))
      (.broadcast (.constFloat .fp32 0) [B])
  | .boolean => .where
      (.ge .nat .scalarR (boolNat B) (.constNat 0))
      (.castFloat .real .fp32 (.natToReal (boolNat B)))
      (.broadcast (.constFloat .fp32 0) [B])
  | .fp16 => .where
      (.ge .fp16 .scalarR (.ref .fp16 [B] "x") (.constFloat .fp16 0))
      (.ref .fp16 [B] "x") (.broadcast (.constFloat .fp16 0) [B])
  | .bf16 => .where
      (.ge .fp32 .scalarR (.castFloat .bf16 .fp32 (.ref .bf16 [B] "x"))
        (.constFloat .fp32 0))
      (.ref .bf16 [B] "x") (.broadcast (.constFloat .bf16 0) [B])
  | .fp32 => .where
      (.ge .fp32 .scalarR (.ref .fp32 [B] "x") (.constFloat .fp32 0))
      (.ref .fp32 [B] "x") (.broadcast (.constFloat .fp32 0) [B])

def storedOp (k : Kind) (B : Nat) : Op .fp32 [B] :=
  match k with
  | .fp16 => .castFloat .fp16 .fp32 (.ref .fp16 [B] "result")
  | .bf16 => .castFloat .bf16 .fp32 (.ref .bf16 [B] "result")
  | _ => .ref .fp32 [B] "result"

def resultValue {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (cell : Cell α) : α :=
  match k with
  | .signed => let v := cell.read .int
      if 0 ≤ v then M.fromInt none v else M.literal none .fp32 0
  | .unsigned => M.fromNat none (cell.read .nat)
  | .boolean => M.fromNat none (if cell.read .bool then 1 else 0)
  | .fp16 => let v := cell.read .fp16
      M.cast none .fp16 .fp32 (if cmp16 (M.literal none .fp16 0) v then v else M.literal none .fp16 0)
  | .bf16 => let v := cell.read .bf16
      M.cast none .bf16 .fp32 (if cmp32 (M.literal none .fp32 0) (M.cast none .bf16 .fp32 v)
        then v else M.literal none .bf16 0)
  | .fp32 => let v := cell.read .fp32
      if cmp32 (M.literal none .fp32 0) v then v else M.literal none .fp32 0

def outputState {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (s : State α) : State α :=
  if s.pids 0 = 0 then
    (TileShape.allIndices [B]).foldl (fun t i =>
      if s.pids 0 * B + i.1.val < N then
        t.write y (s.pids 0 * B + i.1.val) (.mk .fp32
          (resultValue M cmp16 cmp32 k (s.mem x (s.pids 0 * B + i.1.val))))
      else t) s
  else s

#specNonCircular resultValue avoiding [VeriTile.Bench.Repairs.ReluTritonKernel.kernel, FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
