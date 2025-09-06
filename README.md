# Robot-Learning-From-First-Principles

A hands-on project combining simulation and real-world control of the SO100 robotic arm using vision, language, and reinforcement learning.

# File Strcuture
```
rlffp/
├── README.md
└── report/
    ├── assets/
    ├── report.md
    └── literature_review.md
```

## Contents

### Literature Review
We examine current state of the art models(SOTA's)  in robot learning:
- **Action Chunking with Transformers (ACT)**: An advanced imitation learning framework that uses transformers and conditional variational autoencoders to predict action sequences from human demonstrations.
- **SmolVLA**: A compact vision-language-action model that adapts pretrained vision-language models for robotic control with efficient deployment.
- **HIL-SERL**: A human-in-the-loop sample-efficient reinforcement learning approach that combines imitation learning with reinforcement learning through human corrective feedback.

### Main Report
Our comprehensive report covers:
- **Reinforcement Learning for Robotic Manipulation**: Core concepts and popular algorithms like Proximal Policy Optimization (PPO) applied to robotic manipulation tasks.
- **Sim-to-Real Transfer Challenges**: An in-depth analysis of the difficulties in transferring policies from simulation to real-world hardware, including camera alignment, visual perception gaps, and hardware calibration issues.
- **Flow Policy Optimization (FPO)**: An innovative approach that combines flow-based generative models with policy gradient methods, enabling more expressive policy representations.
- **Implementation Details**: Our practical implementation of FPO for the PickCube task with the SO100 robotic arm.
[PPO and FPO Implementation Deployed on SO100 arm](https://github.com/vruga/lerobot-sim2real)
- **Performance Comparison**: Evaluation results comparing FPO and PPO approaches in both simulation and real-world settings.

For detailed information, please refer to the full documents:
- [Literature Review](report/literature_review.md)
- [Main Report](report/report.md)
- [SRA-VJTI Blog Post : From PPO to FPO- Flow Models for Better Policies](https://blog.sravjti.in/2025/08/14/flow-models-for-better-policies.html)
