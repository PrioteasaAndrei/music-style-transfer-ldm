# Musical Style Transfer Using Latent Diffusion Models - LaTeX Report

This directory contains the LaTeX source files for the project report on musical style transfer using latent diffusion models.


## Work

Needed figures:

- loss functions for the autoencoder and style and compression
- reconstructed specgorams just from the encoeder-decoder
- ddim generated specgorams (plot with timesteps)
- 

Andrei

- [ ] architecture
- [ ] experiments

Theo

- [ ] data preprocesing
- [ ] motivation


## Project Structure

```
report/
├── main.tex           # Main LaTeX document
├── introduction.tex   # Introduction section
├── methodology.tex    # Methodology section
├── implementation.tex # Implementation section
├── results.tex       # Results section
├── discussion.tex    # Discussion section
├── conclusion.tex    # Conclusion section
├── references.bib    # Bibliography file
└── README.md         # This file
```

## Building the Report

To build the report, you need a LaTeX distribution installed on your system. The recommended distribution is TeX Live.

### Prerequisites

1. Install TeX Live (if not already installed):
   - On macOS: `brew install basictex`
   - On Ubuntu/Debian: `sudo apt-get install texlive-full`
   - On Windows: Download and install from [TeX Live website](https://tug.org/texlive/)

2. Install required LaTeX packages:
   ```bash
   tlmgr install latexmk
   tlmgr install collection-fontsrecommended
   tlmgr install collection-latexrecommended
   ```

### Building

1. Navigate to the report directory:
   ```bash
   cd report
   ```

2. Build the report using latexmk:
   ```bash
   latexmk -pdf main.tex
   ```

   This will generate `main.pdf` in the same directory.

### Alternative Build Method

If you prefer using pdflatex directly:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Troubleshooting

If you encounter any issues:

1. Make sure all required LaTeX packages are installed
2. Check the log files for specific error messages
3. Ensure all source files are in the correct directory
4. Verify that the bibliography file is properly formatted

