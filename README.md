go read Wan2.1's readme. Install imageio-ffmpeg, and SAM (and pray).


python3 generate.py --task t2v-1.3B --size 832*480 --frame_num 33 --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --clips_dir ./clips --num_videos 10 --prompts "A cup pouring water into a glass, realistic, darker background" "A kitchen sink faucet running, water flowing down the drain, higher contrast" --ckpt_dir ./Wan2.1-T2V-1.3B
