import bench.tritonbench_g.llama_ff_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

theorem prelude_half (p : Parameters) : prelude p=prelude {p with byte:=Bool.false} := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> rfl

def prefixExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    {t : State α // run M (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [prelude_half]
  simp [prelude, source, Source.kernel, Source.half,
      ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, evalDeltas, numeric, bop, natLt, numericLt,
      ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
      integralFloorDiv, integralMod]
  rfl

def columnBlocks (p : Parameters) : Nat := (p.columns+15)/16
def rowBlock {α : Type} (p : Parameters) (s : State α) : Nat := s.pids 0 / columnBlocks p
def columnBlock {α : Type} (p : Parameters) (s : State α) : Nat := s.pids 0 % columnBlocks p
def inputRow {α : Type} (p : Parameters) (s : State α) (i : Fin 16) :=
  (rowBlock p s * 16 + i.val) % p.rows
def inputColumn {α : Type} (p : Parameters) (s : State α) (i : Fin 16) :=
  (columnBlock p s * 16 + i.val) % p.columns
def aOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [16,64]) :=
  inputRow p s i.1 * p.inner + (step*64+i.2.1.val)
def wOffset {α : Type} (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [64,16]) :=
  (step*64+i.1.val) + inputColumn p s i.2.1*p.inner

structure Indices {α : Type} (p : Parameters) (initial : State α)
    (step : Nat) (current : State α) : Prop where
  memory : current.mem = initial.mem
  m : current.regs .int [] "M64" = some (fun _ => (p.rows : Int))
  n : current.regs .int [] "N64" = some (fun _ => (p.columns : Int))
  k : current.regs .int [] "K64" = some (fun _ => (p.inner : Int))
  sak : current.regs .int [] "sak64" = some (fun _ => 1)
  w1k : current.regs .int [] "sw1k64" = some (fun _ => 1)
  w3k : current.regs .int [] "sw3k64" = some (fun _ => 1)
  rms : current.regs .int [] "srms64" = some (fun _ => 1)
  outm : current.regs .int [] "soutm64" = some (fun _ => (p.columns : Int))
  outn : current.regs .int [] "soutn64" = some (fun _ => 1)
  pidm : current.regs .int [] "pid_m" = some (fun _ => (rowBlock p initial : Int))
  pidn : current.regs .int [] "pid_n" = some (fun _ => (columnBlock p initial : Int))
  offk : current.regs .nat [64] "offs_k" = some (fun i => i.1.val)
  aptrs : current.regs .ptr [16,64] "a_ptrs" = some (fun i => (("A" : RegionName),aOffset p initial step i))
  bptrs : current.regs .ptr [64,16] "w1_ptrs" = some (fun i => (("W1" : RegionName),wOffset p initial step i))
  cptrs : current.regs .ptr [64,16] "w3_ptrs" = some (fun i => (("W3" : RegionName),wOffset p initial step i))
  rptrs : current.regs .ptr [1,64] "rms_w_ptrs" = some (fun i : TileIndex [1,64] => (("RMS" : RegionName),step*64+i.2.1.val))

theorem prefixIndices {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    Indices p s 0 (prefixExecution M p s).val := by
  have hc : ((p.columns : Int)+16-1).tdiv 16 = (columnBlocks p : Int) := by
    simpa [columnBlocks, Nat.add_sub_assoc] using signed_cdiv p.columns 16 (by decide)
  have hc' : ((p.columns : Int)+16-1)/16 = (columnBlocks p : Int) := by
    rwa [Int.tdiv_eq_ediv_of_nonneg (by omega)] at hc
  have hn : ¬(p.columns : Int)+16<1 := by omega
  have hp : ¬(s.pids 0 : Int)<0 := by omega
  constructor <;>
    simp [prefixExecution, hc, aOffset, wOffset, inputRow, inputColumn,
      rowBlock, columnBlock, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals try (congr 1; funext i)
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    hc', hn, hp, ← Int.emod_def, ← Int.natCast_ediv,
    ← Int.natCast_mul, ← Int.natCast_add, ← Int.natCast_emod,
    -Nat.cast_mul, -Nat.cast_add, -Int.natCast_mod]
  all_goals
    have rem (x y : Nat) : (x : Int)-(y*(x/y) : Nat)=(x%y : Nat) := by
      simp only [Int.natCast_mul, Int.natCast_ediv, ← Int.emod_def, Int.natCast_mod]
    have nonneg (x l : Nat) : ¬(x : Int)*16+(l : Int)<0 := by omega
    have modTile (x l y : Nat) : ((x : Int)*16+(l : Int))%(y : Int)=((x*16+l)%y : Nat) := by
      simp only [Int.natCast_mod,Int.natCast_add,Int.natCast_mul,Nat.cast_ofNat]
    simp only [rem,nonneg,ite_false,← Int.emod_def,modTile]
    try norm_cast

#axiomsClean prefixIndices
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
