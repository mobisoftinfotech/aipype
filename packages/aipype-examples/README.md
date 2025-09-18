# aipype-examples

Examples and tutorials for the aipype framework.

**Alpha Release** - This package is currently in Alpha stage (v0.1.0a1). APIs may change between releases.

## Overview

This package provides examples and tutorials for the `aipype` framework. It demonstrates how to build AI agents using declarative pipeline-based task orchestration.

## Contents

- **Tutorial**: Step-by-step introduction to aipype concepts
- **Examples**: Real-world usage patterns and agent implementations

## Usage

```python
# Basic example
from aipype_examples.tutorial import OutlineAgent

agent = OutlineAgent(name="example", config={"topic": "AI agents"})
agent.run()
agent.display_results()
```

## Installation

```bash
pip install aipype aipype-examples
```

## Documentation

For complete documentation and getting started guide, see the main [aipype README](../../README.md).

## Status

This is an alpha release. Features and APIs are subject to change.