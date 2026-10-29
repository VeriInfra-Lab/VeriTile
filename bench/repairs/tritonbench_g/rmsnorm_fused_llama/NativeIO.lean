import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Native
import VeriTile.Triton.Float.ObservedProgramEquivalence

/-! Complete canonical row schedule. FP32 source loads, accumulation and
runtime epsilon are retained, including the source FP16 output round trip.
The candidate commutes only an integer pointer product. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeIO
open VeriTile Triton FP.Structural
noncomputable section

def source (candidate : Bool) (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  if candidate then runtimeCandidate X Y W N B else runtimeKernel X Y W N B

def launch (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (row : Nat) : Invocation α where
  kernel := source candidate X Y W N B
  backend := A
  arguments := [⟨"eps",.real,[],fun _ => epsilon⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then row else 0
  numPids := some (fun ax => if ax=0 then rows else 1)

def schedule (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (count : Nat) : List (Invocation α) :=
  (List.range count).map (launch A candidate X Y W rows N B epsilon)

def grid (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) : ProgramIO α α where
  inputs := [⟨.real,[rows,N],X⟩,⟨.real,[N],W⟩]
  outputs := [⟨.real,[rows,N],Y⟩]
  scratch := []
  buffers := [X,W,Y]
  read := fun i => Fin.cases (fun _ j => j.1.val*N+j.2.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.1.val*N+j.2.1.val) (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => schedule A candidate X Y W rows N B epsilon rows
  encode := id
  observe := fun _ _ => id

def commuted (A : Algebra α) (X Y W : RegionName) (rows N B : Nat) (epsilon : α) :
    ProgramIO.Variant (grid A Bool.false X Y W rows N B epsilon) where
  schedule := fun _ => schedule A Bool.true X Y W rows N B epsilon rows
  observe := fun _ _ => id

def original (R : Native.Reductions) (X Y W : RegionName) (rows N : Nat)
    (epsilon : Native.Word) : ProgramIO Native.Term Native.Term :=
  grid (Native.algebra R) Bool.false X Y W rows N 16384 (.leaf "epsilon" 0 epsilon)

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeIO
