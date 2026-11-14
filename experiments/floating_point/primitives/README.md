# Primitive experiments

The [two-gates runner](../README.md) tests atomic scalar expression pairs with
fixed input budgets and independent CPU replay. Typed-count profiles are part
of the default inventory and can also be selected directly:

```bash
python3 scripts/check_two_gates.py check --catalog counts
python3 scripts/check_two_gates.py run --catalog counts --output Logs/two-gates-counts
python3 scripts/check_two_gates.py import Logs/two-gates-counts --output Logs/two-gates-counts.json
```

## Integer-count relations

COUNT-ZERO compares an int32 literal zero converted to fp32 with an fp32 literal
zero. Constant folding is recorded in PTX. Repeated evaluations of a constant
expression do not add stochastic coverage.

COUNT-SUCCESSOR compares `(i+1).to(fp32)` with `i.to(fp32)+1` for uniform int32
inputs `0 <= i < 2^24`. The reference adds in int32 before conversion; the
candidate converts first and adds in fp32. The bound is part of the rule contract:
at `i=16,777,217` the results are 16,777,218 and 16,777,216.

[CountConversion](../../../VeriTile/Triton/Float/CountConversion.lean) binds the
bounded conversion fragment. Using the successor relation requires
`i < upperExclusive`; Welford and LayerNorm therefore require `N <= 2^24` for
row counts. Welford also requires a nonempty row. The experiment sample budget
counts input groups and is independent of this bound on integer values.

The distribution and source precisions are in [count_config.py](count_config.py).
Sampling budgets and gates are in [two_gates_config.py](../two_gates_config.py).
The boundary checker exhaustively evaluates the allowed integer values on GPU:

```bash
python3 scripts/check_count_boundaries.py --output Logs/count-boundaries.json
```

## Exponential relation

EXP-SUB-INTRINSIC compares `tl.exp(a-b)` with `tl.exp(a)/tl.exp(b)` in fp32;
EXP-SUB uses `libdevice.exp`. Each rule has its own source identity and result.
Operands follow the declared `Normal(1,1)` distribution.

```bash
python3 scripts/check_two_gates.py run --catalog supplement --rules EXP-SUB-INTRINSIC --formats fp32 --output Logs/two-gates-exp
```

## Evidence and proof scope

A report binds source casts, input bounds, distribution, backend, PTX and gate
statistics. CPU replay checks the complete saved statistics and required raw
audits. Numerical acceptance applies to that exact rule instance; it is not an
admission of a whole recurrence or reduction. Lean rule records require matching
precision, input domain and evidence identity.
