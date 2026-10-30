# Spec sheet — `bench/tritonbench_g/swiglu_backward/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/swiglu_backward/swiglu_backward.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete source equivalence includes both flag outcomes, ordered stores,
all buffer aliases, masks and failure. Only natural row-address products commute;
no floating operation changes and no numerical assumptions are required. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input upstream gradient output : Storage)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    program input upstream gradient output Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R ≡[assumptions]
      program input upstream gradient output Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`, `template`, `typedLoad`, `typedStore`, `recomputedValue`, `loadAddress`, `storeAddress`

<details><summary><code>program</code></summary>

```lean
def program (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) : Program where
  kernel := typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
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
