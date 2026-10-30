import bench.tritonbench_g.layer_norm_fwd.IO
import VeriTile.Triton.Float.OrdinaryProgramIO

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
variable {α : Type} [Inhabited α]

abbrev inputPorts (r : Regions) (M N : Nat) : List IOChannel :=
  [⟨.real,[M*N],r.x⟩,⟨.real,[M*N],r.x1⟩,⟨.real,[M*N],r.residual⟩,
   ⟨.real,[N],r.weight⟩,⟨.real,[N],r.bias⟩,⟨.real,[N],r.weight1⟩,⟨.real,[N],r.bias1⟩,
   ⟨.real,[2*M],r.rowscale⟩,⟨.int,[2*M],r.seeds⟩]
abbrev outputPorts (r : Regions) (M N : Nat) : List IOChannel :=
  [⟨.real,[M*N],r.y⟩,⟨.real,[M*N],r.y1⟩,⟨.real,[M],r.mean⟩,⟨.real,[M],r.rstd⟩,
   ⟨.real,[M*N],r.residualOut⟩,⟨.bool,[M*N],r.dropoutMask⟩,⟨.bool,[M*N],r.dropoutMask⟩]

/-- Flatten the one-dimensional native channel index for the shared windows. -/
def inputIndex (r : Regions) (M N : Nat) (port : Fin (inputPorts r M N).length) :
    TileIndex (inputPorts r M N)[port].shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ | ⟨6,_⟩ | ⟨7,_⟩ | ⟨8,_⟩ => fun j=>j.1.val

def outputIndex (r : Regions) (M N : Nat) (port : Fin (outputPorts r M N).length) :
    TileIndex (outputPorts r M N)[port].shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ | ⟨6,_⟩ => fun j=>j.1.val

def invocation (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row : Nat) : Invocation α where
  kernel:=quoted f r stride M N eps p
  backend:=engine A
  pids:=fun ax=>if ax=0 then row else 0
  numPids:=some (fun ax=>if ax=0 then M else 1)

/-- Complete same-initial-state ordinary grid. The source-defined row write
sets are proved disjoint; optional ports retain their actual masks and strides. -/
def io (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) : OrdinaryProgramIO α α where
  inputs:=inputPorts r M N
  outputs:=outputPorts r M N
  scratch:=[]
  buffers:=[r.x,r.x1,r.residual,r.weight,r.bias,r.weight1,r.bias1,r.rowscale,r.seeds,
    r.y,r.y1,r.mean,r.rstd,r.residualOut,r.dropoutMask]
  read:=fun port _ j=>RealInterface.inputAddress stride N port (inputIndex r M N port j)
  inputMask:=fun port _ j=>RealInterface.inputActive f M port (inputIndex r M N port j)
  write:=fun port _ j=>RealInterface.outputAddress stride M N port (outputIndex r M N port j)
  outputMask:=fun port _ _=>RealInterface.outputActive f port
  scratchWrite:=fun port=>Fin.elim0 port
  scratchMask:=fun port=>Fin.elim0 port
  schedule:=fun _=>List.ofFn (fun row : Fin M=>invocation A f r stride M N eps p row.val)
  encode:=id
  observe:=fun _ _=>id
  writes:=fun _ row=>Contract.written f r stride M N row.val

abbrev Arrays (r : Regions) (M N : Nat) (α : Type) := ChannelValues α (inputPorts r M N)

def lookup {δ : Type} [Inhabited δ] (a : Fin n → δ) (j : Nat) : δ :=
  if h:j<n then a ⟨j,h⟩ else default

/-- Independent logical arrays, before any source load cast. -/
structure Inputs (α : Type) where
  (x x1 residual : Nat → Nat → α)
  (weight bias weight1 bias1 rowscale : Nat → α)
  seeds : Nat → Int

def logicalInputs (a : Arrays r M N α) : Inputs α where
  x:=fun row col=>lookup (fun j=>a 0 (j,())) (row*N+col)
  x1:=fun row col=>lookup (fun j=>a 1 (j,())) (row*N+col)
  residual:=fun row col=>lookup (fun j=>a 2 (j,())) (row*N+col)
  weight:=lookup (fun j=>a 3 (j,()))
  bias:=lookup (fun j=>a 4 (j,()))
  weight1:=lookup (fun j=>a 5 (j,()))
  bias1:=lookup (fun j=>a 6 (j,()))
  rowscale:=lookup (fun j=>a 7 (j,()))
  seeds:=lookup (fun j=>a 8 (j,()))

def loaded (A : Algebra α) (N : Nat) (values : Nat → α) (col : Nat) : α :=
  TypedSource.cast A (if col<N then values col else TypedSource.lit A 0)

def stream (A : Algebra α) (cmp : TypedSource.Comparisons A) (f : Flags)
    (values : Inputs α) (row : Nat) (p : ℝ) (v : Values α .real [block]) : Values α .real [block] := fun j=>
  let scaled:=if f.rowscale then TypedSource.op A .mul (v j) (TypedSource.cast A (values.rowscale row)) else v j
  if f.dropout then
    if TypedSource.keepValue A cmp (values.seeds row) p j.1.val
    then TypedSource.op A .div scaled (TypedSource.op A .sub (TypedSource.lit A 1) (TypedSource.lit A p))
    else TypedSource.lit A 0
  else scaled

def combined (A : Algebra α) (cmp : TypedSource.Comparisons A) (f : Flags)
    (values : Inputs α) (M N : Nat) (p : ℝ) (row block : Nat) : Values α .real [block] :=
  let first:=stream A cmp f values row p (fun j=>loaded A N (values.x row) j.1.val)
  let second:=stream A cmp f values (M+row) p (fun j=>loaded A N (values.x1 row) j.1.val)
  let both:=fun j=>if f.x1 then TypedSource.op A .add (first j) (second j) else first j
  fun j=>if f.residual then TypedSource.op A .add (both j) (loaded A N (values.residual row) j.1.val) else both j

/-- Original opaque source arithmetic, including masked padding operations,
load/store casts, reduction order, both dropout streams and zero-column stats. -/
def target (A : Algebra α) (cmp : TypedSource.Comparisons A) (f : Flags)
    (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) (width : N≤tile N)
    (a : Arrays r M N α) : ChannelValues α (outputPorts r M N) :=
  let inputs:=logicalInputs a
  let values:=fun row=>combined A cmp f inputs M N p row (tile N)
  fun port j=>match port with
  | ⟨0,_⟩ => TypedSource.affineValue A f.bias
      (TypedSource.normalizedValues A f.rms N eps (values (j.1.val/N))
        (⟨j.1.val%N,lt_of_lt_of_le (Nat.mod_lt _ (by have:=j.1.isLt; nlinarith)) width⟩,()))
      (loaded A N inputs.weight (j.1.val%N)) (loaded A N inputs.bias (j.1.val%N))
  | ⟨1,_⟩ => TypedSource.affineValue A f.b1
      (TypedSource.normalizedValues A f.rms N eps (values (j.1.val/N))
        (⟨j.1.val%N,lt_of_lt_of_le (Nat.mod_lt _ (by have:=j.1.isLt; nlinarith)) width⟩,()))
      (loaded A N inputs.weight1 (j.1.val%N)) (loaded A N inputs.bias1 (j.1.val%N))
  | ⟨2,_⟩ => TypedSource.cast A (TypedSource.meanValue A N (values j.1.val))
  | ⟨3,_⟩ => TypedSource.cast A (TypedSource.rstdValue A f.rms N eps (values j.1.val))
  | ⟨4,_⟩ => TypedSource.cast A (values (j.1.val/N)
      (⟨j.1.val%N,lt_of_lt_of_le (Nat.mod_lt _ (by have:=j.1.isLt; nlinarith)) width⟩,()))
  | ⟨5,_⟩ => TypedSource.keepValue A cmp (inputs.seeds (j.1.val/N)) p (j.1.val%N)
  | ⟨6,_⟩ => TypedSource.keepValue A cmp (inputs.seeds (M+j.1.val/N)) p (j.1.val%N)

#specNonCircular target avoiding [FP.Structural.exec,FP.Structural.run,quoted,TypedSource.oracle]
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
