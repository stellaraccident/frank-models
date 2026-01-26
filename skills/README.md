# Skills Documentation

This directory contains task-oriented documentation for AI agents working with frank-models.

## Skill Categories

### instantiation/

Skills for creating specialized MLIR from templates.

- **rms_norm.md** - Instantiate RMSNorm for specific models

### oracle_generation/

Skills for creating NumPy reference implementations from MLIR.

- **from_mlir.md** - Generate numpy oracle from MLIR component

## How to Use

1. **Identify the task** you're trying to accomplish
2. **Find the relevant skill** in the appropriate category
3. **Follow the process** step by step
4. **Use the validation checklist** to verify correctness

## Adding New Skills

When adding a new skill:

1. Place it in the appropriate category directory
2. Follow this structure:
   - Task Description
   - Input (what you need to start)
   - Output (what you should produce)
   - Process (step-by-step)
   - Example (concrete worked example)
   - Common Mistakes
   - Validation Checklist
