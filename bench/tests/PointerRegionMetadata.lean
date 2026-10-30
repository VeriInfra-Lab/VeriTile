import VeriTile.Triton.DSL

/-! Region metadata follows intermediate pointers without changing the emitted
statements. In particular, a directly quoted transpose retains both ports. -/
namespace PointerRegionMetadataTests
open Lean VeriTile.Triton

private def checkRegions (body : TSyntax `term) (inputs outputs : List String) :
    Elab.Command.CommandElabM Unit := do
  let `(triton { $stmts:tritonStmt* }) := body
    | throwError "expected a triton block"
  let (actualInputs, actualOutputs) := DSL.Metadata.blockRegions stmts.toList
  let names := fun (regions : List (TSyntax `term)) =>
    regions.map (fun region => region.raw.getId.eraseMacroScopes.toString)
  unless names actualInputs == inputs && names actualOutputs == outputs do
    throwError "unexpected region metadata: {names actualInputs}, {names actualOutputs}"

run_cmd do
  checkRegions (← `(triton {
    x = tl.load(A + 1)
    tl.store(O + 1, x)
  })) ["A"] ["O"]
  checkRegions (← `(triton {
    p = A + 1
    out_ptr = O + 2
  })) [] []
  checkRegions (← `(triton {
    p = A + 1
    q = p + 2
    out_ptr = O + 3
    x = tl.load(q)
    tl.store(out_ptr, x)
  })) ["A"] ["O"]
  checkRegions (← `(triton {
    p = A + 1
    p = B + 2
    x = tl.load(p)
  })) ["B"] []
  checkRegions (← `(triton {
    p = A + 1
    p += 2
    p -= 1
    x = tl.load(p)
  })) ["A"] []
  checkRegions (← `(triton {
    if true { p = A + 1 } else { p = B + 2 }
    x = tl.load(p)
  })) ["A", "B"] []
  checkRegions (← `(triton {
    p = A + 1
    if true { p = B + 2 }
    x = tl.load(p)
  })) ["B", "A"] []
  checkRegions (← `(triton {
    p = A + 1
    for i in range(2) {
      q = p + i
      x = tl.load(q)
      tl.store(O + i, x)
    }
  })) ["A"] ["O"]
  checkRegions (← `(triton {
    x = tl.load(A)
    x = x + 1
    tl.store(O, x)
  })) ["A"] ["O"]

def transpose (M Out : RegionName) (m d : Nat) : ComputeKernel := triton {
  tl.region M = tl.float16, Out = tl.float16
  rows = tl.arange(0, $(m))
  cols = tl.arange(0, $(d))
  input_ptr = M + rows[:, None] * $(d) + cols[None, :]
  output_ptr = Out + cols[:, None] * $(m) + rows[None, :]
  values = tl.load(input_ptr)
  tl.store(output_ptr, tl.trans(values))
}

example (M Out : RegionName) (m d : Nat) :
    transpose M Out m d = .mk [M] [Out] (transpose M Out m d).surfaceBody := rfl

end PointerRegionMetadataTests
