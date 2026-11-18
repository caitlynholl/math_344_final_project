Math 334: Cameraman and Barbara denoising application

1. Open run_denoising.m in MATLAB and click Run.
   The images are already included. Only base MATLAB is required.
2. The results folder will contain two comparison PNGs, a CSV table,
   and a LaTeX table with the actual MATLAB results.
3. Upload the results folder to Overleaf. Add \usepackage{graphicx}
   to the preamble if missing.
4. Paste denoising_section.tex as the new Applications subsection,
   or use \input{denoising_section} from your main document.
5. Add reference_to_add.tex inside the existing thebibliography
   environment. If you use BibTeX, create the corresponding entry
   with key zhang2017beyond instead.

The noisy inputs and metric calculations are not clipped. Display range is
[0,1]. Noise standard deviation is 25/255, lambda values are 0.2, 0.7, 1.5,
and lambda is SQUARED in the objective, matching the current project draft.
Barbara is reduced by 2x2 block averaging before noise addition.
Only interior pixel-neighbor differences are penalized (no wrap-around).

Source for the unaltered image files:
https://github.com/cszn/DnCNN/tree/master/testsets/Set12
Cameraman = 01.png; Barbara = 09.png.
Paper: https://arxiv.org/abs/1608.03981 (Figure 3 and Table III).
Image source credits do not imply that the DnCNN authors took the photographs.

Validation: MATLAB is not installed in the authoring environment. The sparse
normal equations and column-major difference operators were checked using
SciPy on both images, including constant preservation and solve residuals.
The MATLAB script must be run to obtain the final experiment's numerical
results. A Python random seed does not reproduce MATLAB's random sequence.
