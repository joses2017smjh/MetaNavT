# Jose Sanchez Gonzalez

I build ML systems and robot perception, with reproducible evaluations.
Seeking **machine learning / AI and robotics engineering roles**.
M.S. Artificial Intelligence and B.S. Computer Science, Oregon State University.

[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) · [Résumé](https://jose-sanchez-portfolio-com.vercel.app/resume.pdf) · [LinkedIn](https://linkedin.com/in/jose-j-sanchez-gonzalez-84a800257/) · [Email](mailto:josejsanchez20172@gmail.com)

[![A simulated pruning arm approaches a branch, performs a gated surrogate release, and returns home](https://raw.githubusercontent.com/joses2017smjh/isaac-sim-pruning-workflow/develop/docs/demo/isaac_two_trees_vision_sequence.gif)](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-pruning-workflow/)

*One controlled Isaac Sim episode. Classical RGB-D tracking, simulator depth and rigid-piece release; this is not a learned pruning policy or a wood-fracture model.*

## Featured work

| Project | What I built | Evidence and limits |
|---|---|---|
| [SPUR metric depth](https://github.com/joses2017smjh/spur-depth-service) | Synthetic depth models, FastAPI inference and split ONNX export. | Encoder parity max difference 1.53e-5; V100 fp16 model p50 156 ms per six-view group. Field accuracy remains unverified. |
| [Humanoid Robustness Ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) | Isaac Lab curricula, ONNX/MuJoCo evaluation and seeded failure controls. | New known-route Isaac task: 379/384 first episodes. Separate frozen-gait MuJoCo inspection: 3/3. No hardware-transfer claim. |
| [Vision-guided pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow) | RGB-D control, dual-ToF release checks and an independent recording grader. | Success and stopped-closure recordings; one selected episode passes 17/17 checks. Known target, classical tracking, rigid-piece release. |
| [MetaNaviT](https://github.com/joses2017smjh/MetaNavT) | Retrieval, data layer and APIs in a six-person team. | Recall@50 0.938 on 136 queries / 61 fixture files, with hash embeddings and overlap reranking. BM25 wins overall nDCG@10. |

## Technical focus

Python, PyTorch, OpenCV, Isaac Sim / Isaac Lab, MuJoCo, ONNX Runtime,
FastAPI and Slurm. I connect model outputs to measurable behavior, document
failure cases, and keep demos linked to the code and evaluation that support them.

The [portfolio](https://jose-sanchez-portfolio-com.vercel.app/#projects) provides
short visual case studies; the repositories contain setup, architecture,
results and current limitations.

[Folding research: policy success/failure and both wrist cameras](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-folding/) — the successful clip is from an earlier checkpoint; the latest adaptation has not shown a gain.
