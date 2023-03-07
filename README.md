# MEMOS
**Mouse Embryo Multi-Organ Segmentations (MEMOS):** A 3D Slicer extension for automated segmentation of fetal mice using deep-learning. For details on the method please see the [open-access paper in Biology Open ](https://journals.biologists.com/bio/article/12/2/bio059698/287076/Deep-learning-enabled-multi-organ-segmentation-of). Step-by-step installation and use instructions, and detailed installation and [usage instructions can be found as a supplementary file accompanying the paper](https://cob.silverchair-cdn.com/cob/content_public/journal/bio/12/2/10.1242_bio.059698/2/bio059698supp.pdf?Expires=1680004103&Signature=UGfnJ4CZw5Tn6w0QPVH-Y35Aj-Mxc~uz6kEIcWjWViL3T~eUp-3MdExKrsp0R2d9PVl8AANduLyoNMXvxYnyIpsYQA3wtpXkPdwP8e-e2OBaUPyAz6Hu2nc8VELVhGSTXOKot0pBO2ATF6vLnPCvwT0VDhglqh-2Rgtl-tdKvv~wp7F9lp3FbfkW1DJ5FuSWrpHL~RY3-o~z02iKb435k~-2lbyW42gspUE~z23pQx6lrXhKSktR-LIjAd5mdFX3fDt6z2owfBDzOEylXRp7aBHU6LJNLpRaSKZKkpKKW2-dEMtUjm0KwjelQx8PgOpG1JAZGBg1HH6VQzKgjXcywg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA).

Briefly you need: 

  1. Install the MEMOs extension. As part of the installation, PyTorch extension will also be installed. Restart Slicer for changes to take effect. 
  2. Switch to pytorch module and choose automatic install (one-time only. You can skip this, if you already installed PyTorch before)
  3. Switch to MEMOS module. 
  4. download the pretrained network (one-time) from [this link](https://app.box.com/shared/static/4nygg33o70oj5xvnhew11zz5geclus5b.pth). 
  5. and obtain a sample dataset [Eg. sample from IMPC](https://raw.githubusercontent.com/SlicerMorph/SampleData/master/IMPC_sample_data.nrrd)
 
If you use this work, please cite **Rolfe SM, Whikehart SM, Maga AM (2023) Deep learning enabled multi-organ segmentation of mouse embryos. Biology Open, 12(2):bio059698. https://doi.org/10.1242/bio.059698**


## Funding
This work was partly supported by grants NIH/OD032627 and NIH/HD104435.

<img src="./memos.jpg">
