import bench.tritonbench_g.context_attn_fwd.Proofs.FPSuccess
import bench.tritonbench_g.context_attn_mistral.Proofs.FPBackend
import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_HostFPComposition

namespace VeriTile.Bench.Repairs.ContextAttnMistral.FPSymbolic
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev program (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ) :=
  VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel p r storage Bool.false scale

def parts (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (program p r storage scale).surfaceBody "start_n" with
  | some q => q
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ) :
    (program p r storage scale).surfaceBody =
      (parts p r storage scale).before ++
      [.forRangeInt "start_n" (parts p r storage scale).start
        (parts p r storage scale).stop (parts p r storage scale).stride
        (parts p r storage scale).body] ++ (parts p r storage scale).after := by
  cases storage <;> rfl

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def Has {α : Type} (s : State α) (d : TileDType) (sh : TileShape) (name : RegName) : Prop :=
  (s.regs d sh name).isSome

def Ready {α : Type} (p : Layout) (storage : Storage) (s : State α) : Prop :=
  Has s .nat [] "cur_head" ∧ Has s .nat [] "start_m" ∧
  Has s .int [] "cur_batch_in_all_start_index" ∧ Has s .int [] "cur_batch_seq_len" ∧
  Has s .nat [p.BM] "offs_m" ∧ Has s .nat [p.BN] "offs_n" ∧ Has s .nat [p.D] "offs_d" ∧
  Has s (storageType storage) [p.BM,p.D] "q" ∧
  Has s .ptr [p.D,p.BN] "k_ptrs" ∧ Has s .ptr [p.BN,p.D] "v_ptrs" ∧
  Has s .real [p.BM] "m_i" ∧ Has s .real [p.BM] "l_i" ∧
  Has s .real [p.BM,p.D] "acc" ∧ Has s .nat [] "block_mask"

/-- Every prelude register is constructed for arbitrary typed memory and
opaque floating values. There is no successful-execution premise. -/
theorem prelude {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ) (s : State α) :
    ∃ t, run (engine o.algebra) (parts p r storage scale).before s = some t ∧
      Ready p storage t ∧ t.mem = s.mem := by
  cases storage
  all_goals
    simp (config := {maxSteps := 300000}) [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile,Source.intRange?,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
      ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
      Ready,Has,storageType]

/-- A complete iteration has all registers, both opaque dots, source arithmetic,
source guards and casts. Every numerical operation remains an opaque callback. -/
theorem body {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ)
    (s : State α) (hBN : 0 < p.BN) (ready : Ready p storage s) (index : Int) :
    ∃ t, run (engine o.algebra) (parts p r storage scale).body
        (s.setReg "start_n" .int [] (fun _ => index)) = some t ∧
      Ready p storage t ∧ t.mem = s.mem := by
  rcases ready with ⟨hh,hb,hs,hl,hr,hk,hd,hq,hkp,hvp,hm,hn,ha,hmask⟩
  obtain ⟨head,hh⟩ := Option.isSome_iff_exists.mp hh
  obtain ⟨block,hb⟩ := Option.isSome_iff_exists.mp hb
  obtain ⟨start,hs⟩ := Option.isSome_iff_exists.mp hs
  obtain ⟨length,hl⟩ := Option.isSome_iff_exists.mp hl
  obtain ⟨rows,hr⟩ := Option.isSome_iff_exists.mp hr
  obtain ⟨keys,hk⟩ := Option.isSome_iff_exists.mp hk
  obtain ⟨dims,hd⟩ := Option.isSome_iff_exists.mp hd
  obtain ⟨query,hq⟩ := Option.isSome_iff_exists.mp hq
  obtain ⟨kptr,hkp⟩ := Option.isSome_iff_exists.mp hkp
  obtain ⟨vptr,hvp⟩ := Option.isSome_iff_exists.mp hvp
  obtain ⟨center,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨normalizer,hn⟩ := Option.isSome_iff_exists.mp hn
  obtain ⟨accumulator,ha⟩ := Option.isSome_iff_exists.mp ha
  obtain ⟨mask,hmask⟩ := Option.isSome_iff_exists.mp hmask
  cases storage
  all_goals simp only [storageType] at hq
  all_goals
    simp (config := {maxSteps := 300000}) [parts,program,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile,Source.intRange?,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
      ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
      Ready,Has,storageType,hh,hb,hs,hl,hr,hk,hd,hq,hkp,hvp,hm,hn,ha,hmask,hBN]

end
end VeriTile.Bench.Repairs.ContextAttnMistral.FPSymbolic
