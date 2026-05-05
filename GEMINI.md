# Gemini Context: tfm-slm

This project is a Master's Thesis (TFM) focused on implementing a **Small Language Model (SLM)** from scratch using a hybrid **Transformer-GRU** architecture and performing a comparative analysis with the **SmolLM** family from Hugging Face.

## Project Structure

- `app/`: Contains the Python source code for the Small Language Model.
  - `main.py`: Entry point for the application.
- `report/`: Contains the LaTeX source for the thesis report.
  - `main.tex`: The main LaTeX file that assembles the document.
  - `content/`: Modular LaTeX files for each chapter (named `1.tex`, `2.tex`, etc.).
  - `NOTOCAR/`: Internal LaTeX configuration, formatting, and structural files. **Do not modify unless changing the report's core style.**
  - `bibliography/`: BibTeX files for citations.
- `tests/`: Directory for project tests.
- `pyproject.toml`: Python project configuration, including Ruff linting settings.

## Research Focus & Methodology

The research addresses the feasibility of custom SLM development versus using established state-of-the-art models like **SmolLM**. The core of the project is a comparative evaluation of two knowledge-infusion strategies applied to both models:

1.  **Retrieval-Augmented Generation (RAG):** Enhancing accuracy by fetching external context from private documentation.
2.  **Fine-tuning via LoRA (Low-Rank Adaptation):** Directly injecting domain-specific knowledge into the model weights.

The evaluation uses a **synthetic Zurich-specific dataset** as the final benchmark to measure performance in specialized domains. The project also focuses on the full engineering lifecycle: CUDA optimization, AWS infrastructure (Spot Instances), and production-grade deployment via Kubernetes and ArgoCD using GitOps.

## Development Workflows

### Python Development
- **Linting:** Run `ruff check .`
- **Formatting:** Run `ruff format .`
- **Execution:** Run `python app/main.py` (Note: Currently a placeholder).

### Report Writing
- Add new content by creating `4.tex`, `5.tex`, etc., in the `report/content/` directory. The main document automatically detects and includes these in numerical order.
- To compile the report, run `pdflatex main.tex` (or your preferred LaTeX compiler) from the `report/` directory.

## Development Conventions

- **SOLID & Clean Code:** All code must adhere to SOLID principles and Clean Code practices to ensure maintainability and readability.
- **Typing Convention:** Use Python 3.10+ built-in generics for type hinting (e.g., `list[int]`, `dict[str, float]`) instead of the `typing` library. All type hints must be in lowercase where applicable.
- **Modular LaTeX:** Keep the report content in `report/content/`. Avoid adding large blocks of text directly to `main.tex`.
- **Ruff Standards:** Adhere to the linting rules defined in `pyproject.toml` (E, F, I, B, S, UP categories).
## Pull Request Guidelines

When creating a PR, the description must follow this structure:

# SUMMARY

 ## Issue type
 ## Component name
 ## Additional information
 ## Validation

And create a .md file with the description.
Do not use emojis and keep the information concise.
Do not include GEMINI.md mention or .geminiignore.
Do not run git comands.
