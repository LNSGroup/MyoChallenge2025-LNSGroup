# MyoChallenge2025-LNSGroup

This repository hosts the submission for **MyoChallenge 2025 Table Tennis Track** by **Team LNSGroup**.

**Challenge webpage:** [MyoChallenge 2025](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025)

## Approach

The competition entry primarily uses **CrossQ**. The setup can be extended toward the latest **Qflex** stack when you want to align with newer algorithms.

## Training code

Algorithm training code lives in the **`Qflex/`** submodule. Use that folder as the reference implementation for training pipelines.

## Quick verification

From the repository root, run the table-tennis agent against the local test environment:

```bash
cd myochallenge_2025eval
sh ./test/test_tabletennis_agent.sh
```

This exercises the gRPC bridge between the CrossQ-based agent and the bundled test environment.
