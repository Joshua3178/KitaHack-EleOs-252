# KitaHack-EleOs-252
Repo Overview: This Repo is a condensed version of a computer Visual Language Model design capable of detecting special unfortunate cases such as bleeding in points during surgery. the basis concept uses Google OWLv2 to propose bounding boxes for two safety-relevant categories: bleeding regions and potentially left-behind surgical equipment. Optionally, it can refine detections with SAM2 segmentation masks, apply tiling for large images, run group-wise NMS, and generate visual overlays plus JSON reports.

Team Introduction
Our team consists of four members:
Ng Yu Meng (Joshua) | Team Leader
Leads overall product development. Responsible for integrating the Vision-Language Model (VLM), conducting testing and evaluation, and supporting model fine-tuning/training experiments.
Wong Xuan Kai | Team Engineer
Responsible for data collection and improving the software structure, including code organization, pipeline stability, and performance refinements.
Liew Xuan Bing | Team Coordinator
Handles documentation and ensures project deliverables are well-prepared, including report writing and proposal elaboration.
Hoe Shan Wa | Team Researcher
Conducts background research and methodology exploration. Supports concept refinement, drafts potential output demonstrations, identifies future development opportunities, and prepares financial-related analysis.

