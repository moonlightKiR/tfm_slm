# SUMMARY
This PR implements the automated CI/CD pipeline for the project's Docker image, optimizes dependency management for containerization, and updates the technical report to document these processes.

## Issue type
Infrastructure / Documentation

## Component name
CI/CD (GitHub Actions), Docker, and Report (LaTeX)

## Additional information
- **Automation:** Created `.github/workflows/docker-image.yml` using Docker Buildx and GitHub Actions cache (gha) for fast and efficient builds. The image is automatically published to GHCR.
- **Dependency Management:** Removed `uv.lock` from `.gitignore` to ensure reproducible builds within the Docker container, as required by the `uv sync` process.
- **Binary Handling:** Created `.gitattributes` to mark `report/main.pdf` as binary, preventing merge conflicts and ensuring consistent overwrites.
- **Documentation:** Updated `report/content/6.tex` with new subsections:
    - "Automatización de la Integración Continua (CI)": Explains the GitHub Actions workflow.
    - "Gestión de Binarios y Control de Versiones": Details the `.gitattributes` strategy for the PDF report.

## Validation
- Verified the GitHub Actions workflow with Buildx setup.
- Confirmed Docker build success locally after tracking `uv.lock`.
- Validated LaTeX consistency and chapter structure.
