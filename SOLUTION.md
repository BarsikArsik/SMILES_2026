# SOLUTION.md

### 1. Zero-order optimization

The baseline skeleton in `zo_optimizer.py` was replaced with an SPSA-style estimator that perturbs all active head parameters simultaneously and estimates the update direction from only two loss evaluations per step, which is much more compute-efficient than perturbing each parameter independently.

The final optimizer tunes only `fc.weight` and `fc.bias`, uses a learning rate of `2e-3`, perturbation size `5e-4`, momentum `0.8`, gradient clipping `1.0`, and gradient normalization enabled, because this region gave the best result in the small grid search over learning rate and epsilon.

In the tuning search, all configurations with `ZO_LR = 0.002` outperformed the smaller tested learning rates, while `ZO_EPS` in the range from `1e-4` to `1e-3` had little effect on the result.

To find this configuration, a separate runner script varied `ZO_LR`, `ZO_EPS` and a few related flags across a small grid and collected `val_accuracy_top1_finetuned` for each combination.

| index | ZO_LR | ZO_EPS | zo_top1 | gain_vs_init | gain_vs_baseline |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.0020 | 0.0005 | 0.0123 | 0.0002 | 0.0086 |
| 6 | 0.0020 | 0.0001 | 0.0123 | 0.0002 | 0.0086 |
| 8 | 0.0020 | 0.0010 | 0.0123 | 0.0002 | 0.0086 |
| 0 | 0.0005 | 0.0001 | 0.0122 | 0.0001 | 0.0085 |
| 1 | 0.0005 | 0.0005 | 0.0122 | 0.0001 | 0.0085 |
| 4 | 0.0010 | 0.0005 | 0.0122 | 0.0001 | 0.0085 |
| 3 | 0.0010 | 0.0001 | 0.0122 | 0.0001 | 0.0085 |
| 2 | 0.0005 | 0.0010 | 0.0122 | 0.0001 | 0.0085 |
| 5 | 0.0010 | 0.0010 | 0.0122 | 0.0001 | 0.0085 |

### 2. Head initialization

The final linear classifier head in `head_init.py` uses Xavier uniform initialization followed by additional scaling of the weights by `0.01`, with zero bias.

This choice provides a conservative initialization for a randomly inserted 100-way classification head on top of a pretrained backbone, keeping initial logits small and helping zero-order fine-tuning remain stable.

In direct comparison against other tested initializations, this modification increased the fine-tuned result from the earlier 1.23% range to 1.24%.

Orthogonal initialization with the same 0.01 scaling was also tried; it significantly reduced both the initialized-head accuracy (to 0.90%) and the fine-tuned result (to 0.69%), so it was discarded.

### 3. Training augmentation

The final training augmentation is intentionally lightweight: `Resize(224)`, `RandomHorizontalFlip()`, a very small `ColorJitter`, then `ToTensor()` and normalization with CIFAR100 statistics.

A broader augmentation sweep showed that heavier spatial or occlusion-based transforms did not improve the metric, while several mild pipelines tied for the best result; the final choice therefore favored the simplest strong option.

| aug_name                   | baseline_top1 | init_top1 | zo_top1 | gain_vs_init |
|----------------------------|--------------:|----------:|--------:|-------------:|
| baseline_plus              | 0.37          | 1.21      | 1.23    | 0.02         |
| crop_flip_jitter_light     | 0.37          | 1.21      | 1.23    | 0.02         |
| crop_flip_jitter_medium    | 0.37          | 1.21      | 1.23    | 0.02         |
| flip_jitter_medium         | 0.37          | 1.21      | 1.23    | 0.02         |
| flip_jitter_tiny           | 0.37          | 1.21      | 1.23    | 0.02         |
| affine_flip_jitter_light   | 0.37          | 1.21      | 1.22    | 0.01         |
| crop_flip                  | 0.37          | 1.21      | 1.22    | 0.01         |
| crop_flip_jitter_erasing   | 0.37          | 1.21      | 1.22    | 0.01         |
| flip_jitter_light          | 0.37          | 1.21      | 1.22    | 0.01         |
| rotate_flip_jitter_light   | 0.37          | 1.21      | 1.22    | 0.01         |

### 4. Training subset selection

The largest improvement came from modifying `train_data.py` to use a balanced subset of the CIFAR100 training split with 50 images per class, for a total of 5,000 training samples.

Under the fixed budget of 32 zero-order steps and batch size 32, this change increased the final fine-tuned accuracy from roughly 1.23–1.24% to 1.54%, while leaving the initialized-head checkpoint unchanged at 1.21%.

This suggests that, in the strict-budget regime, concentrating updates on a smaller, balanced subset is more effective than spreading the same number of steps over the full 50,000-image training split.



## Final configuration

| Component | Final choice | Why it was selected |
|---|---|---|
| `zo_optimizer.py` | SPSA-style simultaneous perturbation, head-only tuning, `lr=2e-3`, `eps=5e-4`, momentum 0.8, grad clip 1.0, normalized updates | Best result in the small ZO hyperparameter search. |
| `head_init.py` | Xavier uniform + weight scaling by 0.01 + zero bias | Best tested head initialization under the official evaluator. |
| `augmentation.py` | Resize + horizontal flip + tiny color jitter + normalization | Tied for the best augmentation result while remaining simpler than crop-heavy alternatives. |
| `train_data.py` | Balanced subset, 50 train samples per class (5,000 total) | Largest measured gain under fixed compute budget. |

## Experiments and failed attempts

### Zero-order hyperparameter search

A 3×3 grid search was run over `ZO_LR ∈ {5e-4, 1e-3, 2e-3}` and `ZO_EPS ∈ {1e-4, 5e-4, 1e-3}` while keeping momentum, clipping, and normalization fixed.

The search showed that `ZO_LR = 2e-3` consistently improved `val_accuracy_top1_finetuned` to 1.23%, while lower learning rates stayed around 1.22%.

No strong sensitivity to epsilon was observed inside the tested interval, so the final solution uses the middle value `5e-4` as a stable default.

```python
import os
import json
import itertools
import subprocess
import pandas as pd

grid = {
    "ZO_LR": [5e-4, 1e-3, 2e-3],
    "ZO_EPS": [1e-4, 5e-4, 1e-3],
    "ZO_MOMENTUM": [0.8],
    "ZO_GRAD_CLIP": [1.0],
    "ZO_NORMALIZE": ["1"],
}

results = []

keys = list(grid.keys())
values = [grid[k] for k in keys]

for combo in itertools.product(*values):
    params = dict(zip(keys, combo))
    run_name = "_".join(f"{k}-{v}" for k, v in params.items()).replace("/", "-")
    output_file = f"{run_name}.json"

    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)

    cmd = [
        "python", "validate.py",
        "--data_dir", "./data",
        "--batch_size", "32",
        "--n_batches", "32",
        "--output", output_file,
    ]

    print(f"\n=== RUN: {run_name} ===")
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)

    row = {**params, "returncode": completed.returncode, "output_file": output_file}

    if completed.returncode != 0:
        row["error"] = completed.stderr[-1500:]
        print("FAILED")
    else:
        try:
            with open(output_file, "r") as f:
                data = json.load(f)

            row["baseline_top1"] = data["val_accuracy_top1_imagenet_head"]
            row["init_top1"] = data["val_accuracy_top1_init_head"]
            row["zo_top1"] = data["val_accuracy_top1_finetuned"]
            row["gain_vs_init"] = row["zo_top1"] - row["init_top1"]
            row["gain_vs_baseline"] = row["zo_top1"] - row["baseline_top1"]
            row["n_batches"] = data["n_batches"]
            row["batch_size"] = data["batch_size"]
            row["total_samples"] = data["total_samples"]
            row["layers_tuned"] = ", ".join(data["layers_tuned"])

            print(
                f"init={row['init_top1']:.4f}, "
                f"zo={row['zo_top1']:.4f}, "
                f"gain={row['gain_vs_init']:+.4f}"
            )
        except Exception as e:
            row["error"] = str(e)
            print("JSON PARSE FAILED:", e)

    results.append(row)

df = pd.DataFrame(results)

if "gain_vs_init" in df.columns:
    df = df.sort_values("gain_vs_init", ascending=False)

df.to_csv("zo_search_summary.csv", index=False)
df
```

### Augmentation ablation

Ten augmentation configurations were compared through repeated official `validate.py` runs, including plain flip, crop-based pipelines, small and medium color jitter, affine transforms, rotation, and random erasing.

The strongest group of pipelines all reached 1.23%, including `baseline_plus`, `flip_jitter_tiny`, `flip_jitter_medium`, `crop_flip_jitter_light`, and `crop_flip_jitter_medium`.

The weaker group reached only 1.22%, including pure crop, affine-based augmentation, rotation-based augmentation, and the erasing-based variant.

Because several pipelines tied at the top, the final solution kept a simple and interpretable training transform rather than adding more aggressive operations that did not improve the score.

```python
import os
import json
import subprocess
import pandas as pd
from textwrap import dedent

AUG_TEMPLATES = {
    "baseline_plus": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "flip_jitter_tiny": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "flip_jitter_light": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "flip_jitter_medium": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "crop_flip": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomCrop(224, padding=28),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "crop_flip_jitter_light": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomCrop(224, padding=28),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "crop_flip_jitter_medium": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomCrop(224, padding=28),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "affine_flip_jitter_light": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomAffine(degrees=0, translate=(0.08, 0.08)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "rotate_flip_jitter_light": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomRotation(degrees=10),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",

    "crop_flip_jitter_erasing": dedent("""
        import torchvision.transforms as T

        _CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        _CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        def get_transforms(train: bool) -> T.Compose:
            if train:
                return T.Compose([
                    T.Resize(224),
                    T.RandomCrop(224, padding=28),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                    T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
                ])
            else:
                return T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                ])
    """).strip() + "\n",
}

if not os.path.exists("augmentation.py.bak"):
    subprocess.run(["cp", "augmentation.py", "augmentation.py.bak"], check=True)

results = []

for aug_name, aug_code in AUG_TEMPLATES.items():
    with open("augmentation.py", "w", encoding="utf-8") as f:
        f.write(aug_code)

    output_file = f"aug_{aug_name}.json"

    cmd = [
        "python", "validate.py",
        "--data_dir", "./data",
        "--batch_size", "32",
        "--n_batches", "32",
        "--output", output_file,
    ]

    print(f"\n=== AUG RUN: {aug_name} ===")
    completed = subprocess.run(cmd, capture_output=True, text=True)

    row = {
        "aug_name": aug_name,
        "returncode": completed.returncode,
        "output_file": output_file,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-1000:],
    }

    if completed.returncode != 0:
        row["error"] = completed.stderr[-1500:]
        print("FAILED")
    else:
        with open(output_file, "r") as f:
            data = json.load(f)

        row["baseline_top1"] = data["val_accuracy_top1_imagenet_head"]
        row["init_top1"] = data["val_accuracy_top1_init_head"]
        row["zo_top1"] = data["val_accuracy_top1_finetuned"]
        row["gain_vs_init"] = row["zo_top1"] - row["init_top1"]
        row["gain_vs_baseline"] = row["zo_top1"] - row["baseline_top1"]

        print(
            f"init={row['init_top1']:.4f}, "
            f"zo={row['zo_top1']:.4f}, "
            f"gain={row['gain_vs_init']:+.4f}"
        )

    results.append(row)

df_aug = pd.DataFrame(results)

if "gain_vs_init" in df_aug.columns:
    df_aug = df_aug.sort_values(
        ["gain_vs_init", "zo_top1", "init_top1", "aug_name"],
        ascending=[False, False, False, True]
    )

df_aug.to_csv("augmentation_search_summary_v2.csv", index=False)

print("\n=== AUG RAW RESULTS ===")
display(df_aug)

if "gain_vs_init" in df_aug.columns:
    view_aug = df_aug.copy()
    for col in ["baseline_top1", "init_top1", "zo_top1", "gain_vs_init", "gain_vs_baseline"]:
        if col in view_aug.columns:
            view_aug[col] = view_aug[col] * 100.0

    print("\n=== AUG TOP RESULTS (percent) ===")
    display(
        view_aug[[
            "aug_name",
            "baseline_top1",
            "init_top1",
            "zo_top1",
            "gain_vs_init",
        ]]
    )
```

### Head initialization trials

Several initialization ideas were considered for the final linear layer, including the original Kaiming-style skeleton, Xavier uniform with small weight scaling, and orthogonal initialization with the same scaling.

Orthogonal + 0.01 performed poorly: the initialized head dropped to 0.90% and the final fine-tuned result dropped further to 0.69%, so this option was discarded.

Xavier uniform + 0.01 gave the best tested behavior, preserving the initialized-head accuracy at 1.21% and raising the fine-tuned result to 1.24%.

### Full train split vs balanced subset

The default full CIFAR100 train split gave results in the 1.23–1.24% range after all other improvements.

A balanced subset with 50 samples per class improved the fine-tuned result to 1.54% while keeping the initialized-head metric unchanged at 1.21%, so the subset strategy was kept in the final solution.

This subset experiment contributed the largest measurable gain among all tested changes and was therefore the most important improvement in the final system.

## Final result summary

Using the final configuration and the official evaluator, the observed checkpoint metrics were approximately: baseline 0.37%, initialized head 1.21%, and fine-tuned zero-order result 1.54%.


