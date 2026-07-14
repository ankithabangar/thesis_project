# Paired t-tests: baseline vs self-reflection

For each (model, metric): each meeting's baseline mean across the 3 runs is paired with its reflection mean, then a two-sided paired t-test is run over the 20 meetings.

Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant, `?` p-value unavailable (scipy not installed).

**Interpretation:** a positive delta means self-reflection improved the metric versus baseline. For higher-is-better faithfulness metrics (AlignScore, MiniCheck, ROUGE, BERTScore, QAFactEval) positive is good.


## AlignScore

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | -0.0014 | -0.13 | 0.9005 | 20 | ns |
| grok | +0.0237 | +2.40 | 0.0268 | 20 | * |
| o4mini | +0.0351 | +2.01 | 0.0591 | 20 | ns |

## MiniCheck mean prob

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | -0.0164 | -0.93 | 0.3630 | 20 | ns |
| grok | +0.0326 | +2.42 | 0.0259 | 20 | * |
| o4mini | +0.0500 | +4.16 | 0.0005 | 20 | *** |

## MiniCheck supported %

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | -0.0328 | -1.49 | 0.1533 | 20 | ns |
| grok | +0.0448 | +2.48 | 0.0225 | 20 | * |
| o4mini | +0.0658 | +3.74 | 0.0014 | 20 | ** |

## QAFactEval

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | +0.0765 | +1.96 | 0.0647 | 20 | ns |
| grok | +0.0449 | +1.85 | 0.0801 | 20 | ns |
| o4mini | +0.0123 | +0.29 | 0.7723 | 20 | ns |

## ROUGE-1

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | +0.0095 | +2.12 | 0.0472 | 20 | * |
| grok | -0.0036 | -1.08 | 0.2937 | 20 | ns |
| o4mini | -0.0060 | -1.81 | 0.0856 | 20 | ns |

## ROUGE-2

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | -0.0001 | -0.07 | 0.9476 | 20 | ns |
| grok | +0.0012 | +1.07 | 0.2980 | 20 | ns |
| o4mini | -0.0003 | -0.18 | 0.8607 | 20 | ns |

## ROUGE-L

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | +0.0075 | +2.95 | 0.0082 | 20 | ** |
| grok | +0.0001 | +0.03 | 0.9734 | 20 | ns |
| o4mini | -0.0006 | -0.32 | 0.7535 | 20 | ns |

## BERTScore-F1

| model  | delta mean | t     | p-value | N (meetings) | sig |
|--------|------------|-------|---------|--------------|-----|
| gemini | +0.0025 | +2.79 | 0.0116 | 20 | * |
| grok | -0.0004 | -0.82 | 0.4237 | 20 | ns |
| o4mini | +0.0003 | +0.48 | 0.6399 | 20 | ns |
