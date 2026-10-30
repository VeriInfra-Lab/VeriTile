# Spec sheet — `bench/tritonbench_g/swiglu_backward/Correct.lean`

**Python source:** `bench/tritonbench_g/swiglu_backward/swiglu_backward.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both full typed variants satisfy independent gradient and optional-forward
formulas from the original input windows, with successful execution and the
complete flat frame. Output regions are pairwise distinct; inputs may alias
outputs because all input loads precede every store. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool)
    (hxy : DX ≠ DY) (hox : OUT ≠ DX) (hoy : OUT ≠ DY) :
    VeriTile.Spec.Real (typedIO input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R ⊨
        fun _ _ xs ys ds => (fun j => Math.dx (xs j) (ys j) (ds j),
          fun j => Math.dy (xs j) (ds j), fun j => Math.forward (xs j) (ys j)))
```

**Assumptions / layout contracts:**
- `hxy : DX ≠ DY`
- `hox : OUT ≠ DX`
- `hoy : OUT ≠ DY`

**Closed-form spec defs (transitive):** `typedIO`, `typedKernel`, `template`, `typedLoad`, `typedStore`, `recomputedValue`, `loadAddress`, `storeAddress`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    Masked2DKernelIO₃ₓ₃ :=
  { swigluBackwardIO X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R with
    kernel := (typedKernel input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R).eraseDType
    projection := typed_supported input upstream gradient output v X Y DOUT OUT DX DY
      sx sy sd so sdx sdy N B R }
```
</details>

<details><summary><code>typedKernel</code></summary>

```
/-- Full wrapper-correlated typed source. The full prefix and fp32 computation
are shared; typed load/store nodes expose all source storage conversions. -/
```
```lean
def typedKernel (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) : ComputeKernel :=
  let body := (template v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody
  ComputeKernel.mk [X,Y,DOUT] [DX,DY,OUT]
    (body.take 9 ++
      [typedLoad "X" "x" input B N, typedLoad "Y" "y" input B N,
       typedLoad "DOUT" "dout" upstream B N] ++
      (body.drop 12).take 3 ++
      [typedStore "DX" "dx" gradient B N, typedStore "DY" "dy" gradient B N,
       .ifThen (.constBool R) [recomputedValue B, typedStore "OUT" "out" output B N]])
```
</details>

<details><summary><code>template</code></summary>

```
/-- Complete source template. Only typed memory-boundary statements are replaced
below; both conditional blocks and all fp32 tensor computations remain present. -/
```
```lean
def template (v : Bool) : Source :=
  if v then swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, commuted]
  else swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, original]
```
</details>

<details><summary><code>typedLoad</code></summary>

```lean
def typedLoad (pointer name : RegName) (input : Storage) (B N : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (loadAddress pointer .real B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [B]))))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (loadAddress pointer .real B N)
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
            ((Op.const 0.0).broadcast [B])))))))
  | .fp16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (loadAddress pointer .fp16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (loadAddress pointer .bf16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))))
  | .f8e4 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e4 .real (.load .f8e4 (loadAddress pointer .f8e4 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B]))))))
  | .f8e5 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e5 .real (.load .f8e5 (loadAddress pointer .f8e5 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B]))))))
```
</details>

<details><summary><code>typedStore</code></summary>

```lean
def typedStore (pointer name : RegName) (storage : Storage) (B N : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp32 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp64 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] name)))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp16 => .store .fp16 [B] (storeAddress pointer .fp16 B)
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [B] name))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .bf16 => .store .bf16 [B] (storeAddress pointer .bf16 B)
      (.alg (.castFloat .real .bf16 (.withPrecision .real .fp32 (.ref .real [B] name))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))

  | .f8e4 => .store .f8e4 [B] (storeAddress pointer .f8e4 B)
      (.alg (.castFloat .real .f8e4 (.withPrecision .real .fp32 (.ref .real [B] name))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .f8e5 => .store .f8e5 [B] (storeAddress pointer .f8e5 B)
      (.alg (.castFloat .real .f8e5 (.withPrecision .real .fp32 (.ref .real [B] name))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
```
</details>

<details><summary><code>recomputedValue</code></summary>

```lean
def recomputedValue (B : Nat) : ComputeStmt :=
  .assign .real [B] "out" (.compute (.alg .fp32
    (.mul .real (Broadcast.consSame .nil)
      (.mul .real (Broadcast.consSame .nil) (.ref .real [B] "x")
        (.ref .real [B] "x_sigmoid")) (.ref .real [B] "y"))))
```
</details>

<details><summary><code>loadAddress</code></summary>

```lean
def loadAddress (pointer : RegName) (dtype : TileDType) (B _N : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd Broadcast.scalarL
    (.ref .ptr [] pointer) (.ref .nat [B] "cols"))
```
</details>

<details><summary><code>storeAddress</code></summary>

```lean
def storeAddress (pointer : RegName) (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [B] "cols"))
```
</details>
