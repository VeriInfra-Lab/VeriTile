# Spec sheet — `bench/tritonbench_g/llama_ff_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/llama_ff_triton/llama_ff_triton.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete state and failure equivalence, including both native-power and
E5M2 backend calls, all casts, two dots, the actual cdiv loop and masked output.
Only the natural row-index product commutes. No floating assumption is used;
all signed int8 payloads, including nonfinite E5M2 encodings, remain in scope. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (fp8 : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    program fp8 Bool.false A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS ≡[assumptions]
      program fp8 Bool.true A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`, `original`, `candidate`, `originalFP8`, `candidateFP8`

<details><summary><code>program</code></summary>

```lean
def program (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) : Program where
  kernel := TypedSource.kernel fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Every source statement for either wrapper weight profile and variant. -/
```
```lean
def kernel (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel :=
  match fp8, variant with
  | Bool.false, Bool.false => original A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
  | Bool.false, Bool.true => candidate A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
  | Bool.true, Bool.false => originalFP8 A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
  | Bool.true, Bool.true => candidateFP8 A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := .mk [A, W1, W3, RMS] [OUT] (triton {
  tl.region A = tl.float16, W1 = tl.float16, W3 = tl.float16, RMS = tl.float16, OUT = tl.float16
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak))
  w1_ptrs = W1 + (offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n))
  w3_ptrs = W3 + (offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n))
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), tl.cdiv($(K), $(BK))) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * $(sak)
    w1_ptrs += $(BK) * $(sw1k)
    w3_ptrs += $(BK) * $(sw3k)
    rms_w_ptrs += $(BK) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + ($(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}).surfaceBody
```
</details>

<details><summary><code>candidate</code></summary>

```lean
def candidate (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := .mk [A, W1, W3, RMS] [OUT] (triton {
  tl.region A = tl.float16, W1 = tl.float16, W3 = tl.float16, RMS = tl.float16, OUT = tl.float16
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak))
  w1_ptrs = W1 + (offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n))
  w3_ptrs = W3 + (offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n))
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), tl.cdiv($(K), $(BK))) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * $(sak)
    w1_ptrs += $(BK) * $(sw1k)
    w3_ptrs += $(BK) * $(sw3k)
    rms_w_ptrs += $(BK) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = $(BM) * pid_m + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + ($(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}).surfaceBody
```
</details>

<details><summary><code>originalFP8</code></summary>

```lean
def originalFP8 (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := .mk [A, W1, W3, RMS] [OUT] (triton {
  tl.region A = tl.float16, W1 = tl.int8, W3 = tl.int8, RMS = tl.int8, OUT = tl.float16
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak))
  w1_ptrs = W1 + (offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n))
  w3_ptrs = W3 + (offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n))
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), tl.cdiv($(K), $(BK))) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs)
    rms_w = rms_w.to(tl.float8e5, bitcast=True)
    rms_w = rms_w.to(tl.float16)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    b = b.to(tl.float8e5, bitcast=True)
    b = b.to(tl.float32)
    b = b.to(tl.float16)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    c = c.to(tl.float8e5, bitcast=True)
    c = c.to(tl.float32)
    c = c.to(tl.float16)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * $(sak)
    w1_ptrs += $(BK) * $(sw1k)
    w3_ptrs += $(BK) * $(sw3k)
    rms_w_ptrs += $(BK) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + ($(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}).surfaceBody
```
</details>

<details><summary><code>candidateFP8</code></summary>

```lean
def candidateFP8 (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := .mk [A, W1, W3, RMS] [OUT] (triton {
  tl.region A = tl.float16, W1 = tl.int8, W3 = tl.int8, RMS = tl.int8, OUT = tl.float16
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak))
  w1_ptrs = W1 + (offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n))
  w3_ptrs = W3 + (offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n))
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), tl.cdiv($(K), $(BK))) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs)
    rms_w = rms_w.to(tl.float8e5, bitcast=True)
    rms_w = rms_w.to(tl.float16)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    b = b.to(tl.float8e5, bitcast=True)
    b = b.to(tl.float32)
    b = b.to(tl.float16)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    c = c.to(tl.float8e5, bitcast=True)
    c = c.to(tl.float32)
    c = c.to(tl.float16)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * $(sak)
    w1_ptrs += $(BK) * $(sw1k)
    w3_ptrs += $(BK) * $(sw3k)
    rms_w_ptrs += $(BK) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = $(BM) * pid_m + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + ($(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}).surfaceBody
```
</details>
