import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_Program
import VeriTile.Triton.Semantics.BroadcastReshape

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
open VeriTile.Triton Math Program
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def parts (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (mathematical p r swapped scale).surfaceBody "start_n" with
  | some x => x
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

def lower (code : List ComputeStmt) : List Stmt :=
  ((ComputeStmt.listToAlgorithm? code).toOption.getD []).map Stmt.eraseDType

def prelude (p : Layout) (r : Regions) : List Stmt :=
  lower (parts p r Bool.false 0).before

theorem prelude_shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (parts p r swapped scale).before = prelude p r := by
  cases swapped <;> rfl

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1,u.pids 2⟩

def inputs (u : BlockState) (r : Regions) : Inputs where
  q := u.readMem r.query
  k := u.readMem r.key
  v := u.readMem r.value
  start := u.readMemValue .int r.start
  sequence := u.readMemValue .int r.sequence
  prompt := u.readMemValue .int r.prompt
  request := u.readMemValue .int r.request
  tokens := u.readMemValue .int r.tokens


def explicitPrelude (p : Layout) (r : Regions) : List Stmt :=
  lower (parts p r Bool.false 0).before

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
