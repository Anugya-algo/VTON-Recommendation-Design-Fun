# VTON-Recommendation-Design-Fun

<a href ="https://drive.google.com/file/d/1snXPdZ_eMX6ID73uuyDHr4rnD5bWsBnv/view?usp=sharing"><a>
Our application enables users to not only visualize garments on themselves but also discover new designs and enjoy gamified shopping with friends.
innovative AI-driven fashion platform combining real-time recommendation, prompt-based outfit generation, and gamified shopping experiences — all built on modern web technologies and intelligent models. The virtual try-on system is under development, using state-of-the-art deep learning for human pose estimation and garment fitting.

https://drive.google.com/file/d/1snXPdZ_eMX6ID73uuyDHr4rnD5bWsBnv/view?usp=sharing

🚀 Features
🔍 AI-Powered Fashion Recommendation

Vector-based search using FashionCLIP

Fast, personalized matching with Pinecone DB

🎮 Fashion Game Mode

Real-time multiplayer outfit styling challenge

Timer, voting, and leaderboard functionality

🛍️ Shopping Experience

Cart-based recommendation system

Modern, responsive React UI

🎨 Prompt-to-Outfit Designer

Describe an outfit in words — get AI-generated visuals

Built with Stable Diffusion v1.5 + LoRA fine-tuned models

👚 Virtual Try-On (Coming Soon)

Using DeepLabV3+ for segmentation

AlphaPose for human pose estimation

HF-VTON pipeline for garment warping and rendering



🧪 Model Checkpoints
Model	Format	Usage
FashionCLIP	.pt	Embedding generation
Stable Diffusion	.ckpt	Outfit generation (with LoRA)
DeepLabV3+	.pth	Human segmentation
AlphaPose	pre-built	Pose detection (keypoints)
HF-VTON	 try-on synthesis
