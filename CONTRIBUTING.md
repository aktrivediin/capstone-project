# Contributing

Thanks for your interest in this project.

This repository was originally developed as an academic capstone submission, and is now maintained
as a portfolio project. Contributions, suggestions, and issue reports are welcome.

## How to contribute

1. Fork the repository.
2. Create a feature branch: git checkout -b feature/your-feature-name.
3. Make your changes, keeping each Part (part1_eda, part2_ml, part3_ensembles, part4_llm)
   independently runnable — a change in one Part's script should not require changes in another
   unless you're also updating the shared cleaned_data.csv / best_model.pkl pipeline outputs.
4. Test that your changed script/notebook still runs top-to-bottom without errors.
5. Update the relevant README.md (root or Part-level) to reflect any change in results or method.
6. Commit your changes with a clear message and open a Pull Request describing what changed and why.

## Code style

- Follow standard PEP 8 conventions for Python code.
- Keep functions documented with a short docstring where behavior isn't obvious.
- Prefer clarity over cleverness — this project favors readable, reproducible steps over dense code.

## Reporting issues

If you find a bug, a reproducibility issue, or an inconsistency between the code and the README,
please open a GitHub Issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce (including your Python version and OS)

## Code of Conduct

Please review CODE_OF_CONDUCT.md before participating.
