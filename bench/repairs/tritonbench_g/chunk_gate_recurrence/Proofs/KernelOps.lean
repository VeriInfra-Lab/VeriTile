import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelOps
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def matrixPtr (name : RegName) : Op .ptr [64,16] := .ref .ptr [64,16] name
def scalarPtr (name : RegName) : Op .ptr [] := .ref .ptr [] name
def matrix (name : RegName) : Op .real [64,16] := .ref .real [64,16] name
def scalar (name : RegName) : Op .real [] := .ref .real [] name
def stride (c : Shape) : Op .nat [] := .mul .nat .nil (.constNat c.K) (.constNat c.V)
def tileStride : Op .nat [] := .mul .nat .nil (.ref .nat [] "NUM_K") (.ref .nat [] "NUM_V")
def update (carry incoming : RegName) : Op .real [64,16] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarR (matrix carry) (scalar "d_i")) (matrix incoming)
def gradient : Op .real [] :=
  .reduceSum ⟨0,by decide⟩ Bool.false (.reshape [64*16]
    (.mul .real (.consSame (.consSame .nil)) (matrix "Dacc")
      (.castFloat .fp32 .real (.ref .fp32 [64,16] "S_i"))))
def assign32 (name : RegName) (op : Op .real [64,16]) : ComputeStmt :=
  .assign .real [64,16] name (.compute (.alg .fp32 op))
def loadMatrix (name pointer : RegName) : ComputeStmt :=
  .assign .real [64,16] name (.compute (.load .fp32 (.ptr (matrixPtr pointer)) .none))
def loadScalar (name pointer : RegName) : ComputeStmt :=
  .assign .real [] name (.compute (.load .fp32 (.ptr (scalarPtr pointer)) .none))
def storeMatrix (pointer value : RegName) : ComputeStmt :=
  .store .real [64,16] (.ptr (matrixPtr pointer))
    (.compute (.alg .fp32 (.castFloat .real .real (matrix value)))) .none
def storeScalar (pointer value : RegName) : ComputeStmt :=
  .store .real [] (.ptr (scalarPtr pointer))
    (.compute (.alg .fp32 (.castFloat .real .real (scalar value)))) .none
def advance (name : RegName) (c : Shape) : ComputeStmt :=
  .assign .ptr [64,16] name (.alg (.ptrAdd .scalarR (matrixPtr name) (stride c)))
def retreat (name : RegName) (c : Shape) : ComputeStmt :=
  .assign .ptr [64,16] name (.alg (.ptrSub .scalarR (matrixPtr name) (stride c)))

def forwardBody (c : Shape) : List ComputeStmt :=
  [loadScalar "d_i" "D",loadMatrix "S_i" "S",assign32 "acc" (update "acc" "S_i"),
   storeMatrix "O" "acc",.assign .ptr [] "D" (.alg (.ptrAdd .nil (scalarPtr "D") (.constNat 1))),
   advance "S" c,advance "O" c]
def backwardBody (c : Shape) : List ComputeStmt :=
  [.assign .fp32 [64,16] "S_i" (.alg (.load .fp32 (.ptr (matrixPtr "S")) .none)),
   loadMatrix "DS_i" "DS",loadScalar "d_i" "D",assign32 "Dacc" (update "Dacc" "DS_i"),
   .assign .real [] "DG_i" (.compute (.alg .fp32 gradient)),
   storeScalar "DG" "DG_i",storeMatrix "DI" "Dacc",
   retreat "S" c,retreat "DI" c,retreat "DS" c,
   .assign .ptr [] "DG" (.alg (.ptrSub .nil (scalarPtr "DG") tileStride)),
   .assign .ptr [] "D" (.alg (.ptrSub .nil (scalarPtr "D") (.constNat 1)))]

theorem forward_body (c : Shape) (s : SeedStrides) (seed : Bool) :
    body .forward c s seed=forwardBody c := by rfl
theorem backward_body (c : Shape) (s : SeedStrides) (seed : Bool) :
    body .backward c s seed=backwardBody c := by rfl

def next (M : Algebra α) (carry : Lane → α) (gate : α) (incoming : Lane → α) : Lane → α :=
  fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (carry i) gate) (incoming i)
theorem update_eval (M : Algebra α) (carry incoming : RegName) (a x : Lane → α) (g : α)
    (u : State α) (ha : u.regs .real [64,16] carry=some a)
    (hx : u.regs .real [64,16] incoming=some x) (hg : u.regs .real [] "d_i"=some (fun _ => g)) :
    FP.Structural.evalOp (engine M) (some .fp32) (update carry incoming) u=some (next M a g x) := by
  simp only [update,matrix,scalar,FP.Structural.evalOp,ha,hx,hg]
  rfl

#axiomsClean forward_body
#axiomsClean backward_body
#axiomsClean update_eval
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelOps
