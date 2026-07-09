go read Wan2.1's readme. Install imageio-ffmpeg, gradio, and SAM (and pray).


python3 generate.py --task t2v-1.3B --size 832*480 --frame_num 33 --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --clips_dir ./clips --num_videos 10 --prompts "A cup pouring water into a glass, realistic, darker background" "A kitchen sink faucet running, water flowing down the drain, higher contrast" --ckpt_dir ./Wan2.1-T2V-1.3B


python annotate.py --clips_dir ./clips --out prompts.json  --redo
python segment.py --clips_dir ./clips --prompts prompts.json --out_dir ./masks --redo
python flow.py --clips_dir ./clips --masks_dir ./masks --out_dir ./flow 
python score_clip.py --masks_dir ./masks --flow_dir ./flow --out_dir ./results

python gradio/score_viewer.py --results_dir ./results
