# MEMOS
**Mouse Embryo Multi-Organ Segmentations (MEMOS):** A 3D Slicer extension for automated segmentation of fetal mice using deep-learning. For details on the method please see the [open-access paper in Biology Open ](https://journals.biologists.com/bio/article/12/2/bio059698/287076/Deep-learning-enabled-multi-organ-segmentation-of). Step-by-step installation and use instructions, and detailed installation and [usage instructions can be found as a supplementary file accompanying the paper](https://cob.silverchair-cdn.com/cob/content_public/journal/bio/12/2/10.1242_bio.059698/2/bio059698supp.pdf?Expires=1680004103&Signature=UGfnJ4CZw5Tn6w0QPVH-Y35Aj-Mxc~uz6kEIcWjWViL3T~eUp-3MdExKrsp0R2d9PVl8AANduLyoNMXvxYnyIpsYQA3wtpXkPdwP8e-e2OBaUPyAz6Hu2nc8VELVhGSTXOKot0pBO2ATF6vLnPCvwT0VDhglqh-2Rgtl-tdKvv~wp7F9lp3FbfkW1DJ5FuSWrpHL~RY3-o~z02iKb435k~-2lbyW42gspUE~z23pQx6lrXhKSktR-LIjAd5mdFX3fDt6z2owfBDzOEylXRp7aBHU6LJNLpRaSKZKkpKKW2-dEMtUjm0KwjelQx8PgOpG1JAZGBg1HH6VQzKgjXcywg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA).

Briefly you need: 

  1. (Optional) install PyTorch extension that will provide torch library most suitable for your platform. This step is necessary for GPU based inference. Otherwise, MEMOS will fallback to CPU based inference. Please install PyTorch prior to MEMOS. 
  2. Install MEMOS extension. 
  3. download the pretrained network (one-time) from [this link](https://app.box.com/shared/static/4nygg33o70oj5xvnhew11zz5geclus5b.pth). 
  4. and obtain a sample dataset [Eg. CBX4 mutant From IMPC](https://www.mousephenotype.org/embryoviewer/?gene_symbol=CBX4)

If you use this work, please cite **Rolfe SM, Whikehart SM, Maga AM (2023) Deep learning enabled multi-organ segmentation of mouse embryos. Biology Open, 12(2):bio059698. https://doi.org/10.1242/bio.059698**


## Funding
This work was partly supported by grants NIH/OD032627 and NIH/HD104435.

<img src="./memos.jpg">
