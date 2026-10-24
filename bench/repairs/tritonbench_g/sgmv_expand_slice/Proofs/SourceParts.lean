import bench.repairs.tritonbench_g.sgmv_expand_slice.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceParts
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000

def tag : ComputeStmt → String × Nat
  | .assign _ _ name _ => (name,0)
  | .ifThen _ body => ("ifThen",body.length)
  | .ifThenElse _ yes no => ("ifThenElse",yes.length+no.length)
  | .forRangeInt name _ _ _ body => (name,body.length)
  | .effectMarker name => (name,0)
  | .store .. => ("store",0)
  | _ => ("other",0)

def thenBody (code : List ComputeStmt) (index : Nat) : List ComputeStmt :=
  match code[index]? with
  | some (.ifThen _ body) => body
  | _ => []

def condition (code : List ComputeStmt) (index : Nat) : Op .bool [] :=
  match code[index]? with
  | some (.ifThen cond _) => cond
  | _ => .constBool Bool.false

def outer (abi : SourceABI) (p : KernelArgs) := (kernel abi p).surfaceBody
def afterLength (abi : SourceABI) (p : KernelArgs) := thenBody (outer abi p) 16
def afterTile (abi : SourceABI) (p : KernelArgs) := thenBody (afterLength abi p) 0
def active (abi : SourceABI) (p : KernelArgs) := thenBody (afterTile abi p) 1
def prelude (abi : SourceABI) (p : KernelArgs) := (outer abi p).take 16
def activePrelude (abi : SourceABI) (p : KernelArgs) := (active abi p).take 7
def suffix (abi : SourceABI) (p : KernelArgs) := (active abi p).drop 8

def loopParts (abi : SourceABI) (p : KernelArgs) : Source.IntRange :=
  match Source.intRange? (active abi p) "k" with
  | some result => result
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩
def body (abi : SourceABI) (p : KernelArgs) := (loopParts abi p).body

theorem kernel_split (abi : SourceABI) (p : KernelArgs) :
    outer abi p = prelude abi p ++
      [.ifThen (condition (outer abi p) 16)
        [.ifThen (condition (afterLength abi p) 0)
          ((afterTile abi p).take 1 ++ [.ifThen (condition (afterTile abi p) 1)
            (activePrelude abi p ++
              [.forRangeInt "k" (loopParts abi p).start (loopParts abi p).stop
                (loopParts abi p).stride (body abi p)] ++ suffix abi p)])]] := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

/-- The actual loop has no output or metadata-width-dependent syntax. This
reduction proves that all those source faces share the very same loop body. -/
theorem body_same_ABI (abi : SourceABI) (p : KernelArgs) :
    body abi p = body {abi with output := .single, startsWidth := .w32, lengthsWidth := .w32, adaptersWidth := .w32} p := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

#axiomsClean kernel_split
#axiomsClean body_same_ABI
#reduce fun p => (body primaryHalf p).map tag

def assignTag : ComputeStmt → Option (RegName × TileDType)
  | .assign d _ name _ => some (name,d)
  | _ => none
def branchTags : ComputeStmt → List (Option (RegName × TileDType))
  | .ifThen _ yes => yes.map assignTag
  | .ifThenElse _ yes no => (yes++no).map assignTag
  | st => [assignTag st]
#reduce fun p => (body primaryHalf p).flatMap branchTags

end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceParts
