#!/bin/bash

# python inference_vm.py --save_dir output/grm --task_json tasks/train/pick_the_red_apple_using_left_arm.json --prompt_idx 0 --inference_prefix debug_6 --tts --ports 23921 23931 23941
# python inference_idm.py --use_image_high --use_image_front --use_image_left --use_image_right --use_image_side --use_puppet_left --use_puppet_right --save_dir output/grm --task_json tasks/train/pick_the_red_apple_using_left_arm.json --inference_prefix debug_5 --video_file tts.mp4 --model_name mask --load_from /home/agilex/fy-cobot-magic/output/0418_mw0.001_60000.pt

task_json=$1 # "/home/agilex/fy-cobot-magic/tasks/ood/wipe_table_with_rag_precise.json"
inference_prefix=$2 # "test_0/1/2"
model=$3 # "grm" grm-resnet grm-without-tts unipi vpp
echo $model
vm_idm=$4 # 1-vm 2-idm 3-vm+idm 4-vpp
# exclude the first four arguments 
shift 4
ports=$@ # "23921 23931 23941" grm 51 61 71 unipi-23981 vpp 23911 grm-nopretrain 23981 23991 24001
echo $ports


if [ "$model" == "dino" ]; then
    model_name="dino"
    save_dir="output/anypos-dino"
    load_from=""
elif [ "$model" == "direction_aware" ]; then
    model_name="direction_aware"
    save_dir="output/anypos-direction_aware"
    load_from=""
elif [ "$model" == "anypos-resnet" ]; then
    model_name="resnet"
    save_dir="output/anypos-resnet"
    load_from=""
elif [ "$model" == "direction_aware_with_split" ]; then
    model_name="direction_aware_with_split"
    save_dir="output/direction_aware_with_split_demo_7_16"
    load_from="/home/agilex/fy-cobot-magic/output/64000.pt"
elif [ "$model" == "resnet_with_split" ]; then
    model_name="resnet_with_split"
    save_dir="output/anypos-resnet_with_split"
    load_from=""
elif [ "$model" == "dino_with_split" ]; then
    model_name="dino_with_split"
    save_dir="output/dino_with_split_demo_7_19"
    load_from=""
elif [ "$model" == "grm" ]; then
    model_name="mask"
    save_dir="output/grm_demo_7_26"
    load_from="/home/agilex/fy-cobot-magic/output/3e-3_60000.pt"
elif [ "$model" == "grm-nopretrain" ]; then
    model_name="mask"
    save_dir="sup/grm-nopretrain"
    load_from="/home/agilex/fy-cobot-magic/output/3e-3_60000.pt"
elif [ "$model" == "vpp" ]; then
    model_name="mask"
    save_dir="output/vpp_demo_7_23"
    load_from="/home/agilex/fy-cobot-magic/output/3e-3_60000.pt"
elif [ "$model" == "grm-without-tts" ]; then
    model_name="mask"
    save_dir="output/grm-without-tts"
    load_from="/home/agilex/fy-cobot-magic/output/3e-3_60000.pt"
# elif [ "$model" == "grm-unet" ]; then
#     model_name="unet"
#     save_dir="output/grm-unet"
#     load_from="/home/agilex/fy-cobot-magic/output/grm-unet.pt"
elif [ "$model" == "grm-resnet" ]; then
    model_name="resnet"
    save_dir="output/grm-resnet"
    load_from="/home/agilex/fy-cobot-magic/output/grm-resnet.pt"
elif [ "$model" == "unipi" ]; then
    model_name="resnet"
    save_dir="sup/unipi"
    load_from="/home/agilex/fy-cobot-magic/output/grm-resnet.pt"
else
    echo "Invalid model name. Please use 'direction_aware_with_split' or 'mask'."
    exit 1
fi

# if [ $vm_idm -eq 1 ] || [ $vm_idm -eq 3 ]; then
#     python inference_camera_prepare.py --use_image_high --use_image_front --use_image_left --use_image_right  --save_dir $save_dir --task_json $task_json --inference_prefix $inference_prefix --video_file tts.mp4
# fi

if [ $vm_idm -eq 1 ] || [ $vm_idm -eq 3 ]; then
    python inference_vm.py --save_dir $save_dir --task_json $task_json --prompt_idx 0 --inference_prefix $inference_prefix --ports $ports --tts
fi

if [ $vm_idm -eq 2 ] || [ $vm_idm -eq 3 ]; then
    python inference_idm_demo.py --use_image_high --use_image_front --use_image_left --use_image_right --use_image_side --use_puppet_left --use_puppet_right --save_dir $save_dir --task_json $task_json --inference_prefix $inference_prefix --video_file tts.mp4 --model_name $model_name --load_from $load_from
fi


if [ $vm_idm -eq 4 ]; then
    python inference_vpp.py --use_image_high --use_image_front --use_image_left --use_image_right --use_image_side --use_puppet_left --use_puppet_right --save_dir "sup/vpp" --task_json $task_json --inference_prefix $inference_prefix --video_file tts.mp4 --model_name $model_name --load_from $load_from
fi

# ./scripts/inference/inference.sh /home/agilex/fy-cobot-magic/tasks/train/stack_one_can_on_top_of_the_other_can_using_right_arm.json test_0 grm 3 23951 23961 23971

# ./scripts/inference/inference.sh /home/agilex/fy-cobot-magic/tasks/ood/pick_the_carrot_and_benana_using_both_arm.json demo_0 grm 3 23951 23961 23971