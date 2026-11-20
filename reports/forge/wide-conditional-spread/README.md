# Wide conditional spread

<a name="wide_conditional_spread"></a>

This Tier 2 stretch task asks whether a GAN learns which group varies along each axis. The two equally likely group labels select zero-mean Gaussian signals with SDs (0.5, 2) and (2, 0.5). The remaining 63 real coordinates are zero. Every generated coordinate comes from the network. A separate label accompanies each generator, discriminator and gradient penalty call.

The frozen task uses 1,200 updates and 24 observations with five terminal passes required. Each observation draws 2,048 independent bags of 32 per group using the live generator and public learned MoG prior. For each signal coordinate, the 10th and 90th percentiles of generated bag means and sample SDs are evaluated on independent target bags. The interval score is width plus ten times the distance outside the interval. The reference score uses another independent sample from the target law. The task gates the worst group/coordinate errors and the summed mean squared output on the 63 inactive coordinates. Training uses protocol seed 0 and its own random streams. The fixed evaluation panels are shared across observations.

The initial adapter binds the public EMPF components with explicit continuous inputs and a penalty on each update. A future comparison can change the conditional discrepancy within EMPF and retain the identical task. The CPU observer regression in [PR #556](https://github.com/HyperGAN/ParticleGAN/pull/556) tests the mechanism. This task scores generated samples. [Task declaration](../../../configs/forge/tasks/wide_conditional_spread.json)

## Released EMPF baseline

On CPU, the released EMPF configuration completed all 1,200 updates with 1,200 penalty applications. The terminal checkpoint reached **65.7% variability coverage and 71.4% combined coverage**, against the 80% target. The worst individual coverage error was 23.4 percentage points, above the 12-point tolerance. The sustained quality verdict was **FAIL**. The generator, critic and prior each updated 1,200 times. [Full metrics and source identity](baseline.json) · [24-frame training GIF](baseline.gif)

To reproduce a direct CPU integration run from the repository root, use the experiment environment and an empty output directory. For ordinary campaign planning use `python -m experiments.forge plan empf-v8 --view discriminator_stability --through-tier 2 --device cpu`.

```python
import json
from pathlib import Path
import torch
from experiments.forge.adapters import run_task
from experiments.forge.views import grade_result

torch.set_num_threads(1)
cards = Path("configs/forge")
task = json.loads((cards / "tasks/wide_conditional_spread.json").read_text())
candidate = json.loads((cards / "ideas/empf-v8.json").read_text())
request = {"candidate": candidate, "tasks": {task["id"]: task}, "protocol": {"seed": 0}}
receipt = run_task(request, {"task_id": task["id"]}, Path("runs/forge/spread-baseline"), "cpu")
print(grade_result(task, receipt)["status"])
```
