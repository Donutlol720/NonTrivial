# Step 6: Transfer-control ambiguity

Transfer rule used throughout this family-level pass:

> Successful transfer requires **negative transfer_effect in absolute terms**.
> Surviving a control comparison (degraded families show more-negative / less-positive transfer than control families) is a consistency check, NOT sufficient to call transfer successful.

### 4 explicit questions

1. **Is the raw mean transfer effect negative for degraded families?**
   - raw (all 5 late layers × family counts pooled) = `-3.1321` → **YES — negative**.

2. **Is the late-layer family-level mean transfer effect negative for degraded families?**
   - family-level degraded mean = `-4.1274` (n=36 families)
   - family-level control mean  = `-1.7454` (n=31 families)
   - → **YES — negative (successful absolute transfer)**.

3. **If transfer survives controls but has the wrong raw sign, what does 'survives' mean?**
   - Expected-signed metric for transfer is defined as `-transfer_effect`. Larger = more degradation transfer.
   - Degraded expected-signed mean = `-(-4.1274) = 4.1274`.
   - Control  expected-signed mean = `-(-1.7454) = 1.7454`.
   - Survives controls (one-sided Mann–Whitney degraded expected-signed > control expected-signed) evaluates: *is degraded transfer directionally more negative (or less positive) than control transfer?*
   - Relative ordering test result: degraded > control in expected-signed metric = **True**.
   - This is a consistency check only. The absolute-value rule takes precedence.

4. **Does transfer provide evidence that pressure activations are sufficient to induce degradation in neutral prompts?**
   - → **YES (absolute transfer effect is negative and statistically distinguishable from 0 or controls)**.


### One-line interpretation of transfer:

Pressure activations, patched from degraded-family pressure prompts into their same-family neutral prompts, shift the neutral evidence-aligned margin **downward** (negative transfer_effect) — supporting sufficiency.
