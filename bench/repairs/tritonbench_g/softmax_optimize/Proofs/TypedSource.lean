import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Source

/-! Typed compiler input. Every fragment is extracted from the actual shared
ComputeKernel, before dtype erasure. The equalities below check the four loop
positions, identical guarded updates, and the two output arithmetic fragments.
The compiler contract in Lowering is local to these extracted expressions. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.TypedSource
open VeriTile.Triton
local instance : Inhabited ComputeStmt := ⟨.effectMarker "invalid.fragment"⟩
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

def body (Y X : RegionName) (N B : Nat) : List ComputeStmt :=
  (kernel Y X 0 N B).surfaceBody

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeInt _ _ _ _ statements => statements
  | _ => []

def fullInput (X : RegionName) (N B : Nat) : List ComputeStmt :=
  loopBody ((body "typed.output" X N B)[5]!)
def tailInput (X : RegionName) (N B : Nat) : List ComputeStmt :=
  loopBody ((body "typed.output" X N B)[6]!)
def fullOutput (Y X : RegionName) (N B : Nat) : List ComputeStmt :=
  loopBody ((body Y X N B)[11]!)
def tailOutput (Y X : RegionName) (N B : Nat) : List ComputeStmt :=
  loopBody ((body Y X N B)[12]!)
def guarded (B : Nat) : List ComputeStmt := (fullInput "typed.input" 1 B).drop 4
def maximum (B : Nat) : List ComputeStmt := ((body "typed.output" "typed.input" 1 B).drop 7).take 1
def weighted (B : Nat) : List ComputeStmt := ((body "typed.output" "typed.input" 1 B).drop 8).take 1
def normalize (B : Nat) : List ComputeStmt := ((fullOutput "typed.output" "typed.input" 1 B).drop 4).take 2

theorem full_guarded (X : RegionName) (N B : Nat) :
    (fullInput X N B).drop 4 = guarded B := by rfl

theorem tail_guarded (X : RegionName) (N B : Nat) :
    (tailInput X N B).drop 5 = guarded B := by rfl

theorem output_normalize (Y X : RegionName) (N B : Nat) :
    ((fullOutput Y X N B).drop 4).take 2 = normalize B ∧
    ((tailOutput Y X N B).drop 5).take 2 = normalize B := by exact ⟨rfl, rfl⟩

/-- Exact typed source decomposition. In contrast to Source.source_body, this
retains all ComputeExpr precision tags, float32 loads, casts and stores. -/
theorem source_body (Y X : RegionName) (M N B : Nat) :
    (kernel Y X M N B).surfaceBody =
      (body Y X N B).take 5 ++
      [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
        (fullInput X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "N") (.constNat B)
        (tailInput X N B)] ++
      ((body Y X N B).drop 7).take 4 ++
      [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
        (fullOutput Y X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "N") (.constNat B)
        (tailOutput Y X N B)] := by rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.TypedSource
