# Does the model know how many times it's called a tool?

The model is asked to call a tool a random number of times (between 1 and 100) and asked to output how many times it called thet tool.

## Results Summary

| Model                    | Experiments | Accuracy | Most Common Calls | Most Common % |
|--------------------------|-------------|----------|-------------------|---------------|
| claude-4-sonnet-20250514 | 10          | 0%       | 61                | 90%           |
| claude-4-opus-20250514   | 10          | 100%     | 42                | 80%           |
| gpt-4o                   | 10          | 70%      | 20                | 30%           |
| gpt-4o-mini              | 10          | 10%      | 75                | 10%           |
| gpt-4.1                  | 10          | 100%     | 10                | 100%          |
| gpt-4.1-mini             | 10          | 60%      | 20                | 30%           |
| o3                       | 10          | 90%      | 9                 | 30%           |
| o4-mini                  | 10          | 50%      | 3                 | 20%           |


## Usage

```bash
# Run with default settings (Claude Sonnet 4, 10 experiments)
uv run experiment.py

# Run with different model
uv run experiment.py --model "openai:gpt-4o-mini"

# Run with more experiments
uv run experiment.py --experiments 50

# Combined options
uv run experiment.py --model "openai:gpt-4o" --experiments 25
```
