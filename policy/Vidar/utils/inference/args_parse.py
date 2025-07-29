import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.', default='assets/debug', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=0, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names', default=['cam_front', 'cam_left_wrist', 'cam_right_wrist', 'cam_high'], required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate', default=30, required=False)
    parser.add_argument('--use_keyboard_end', action='store_true', help='use_keyboard_end', default=False, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.', default=50000, required=False)
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=False, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--max_pos_lookahead', action='store', type=int, help='max_pos_lookahead',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=30, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)
                        # default=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.1], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true', help='use_actions_interpolation', default=False, required=False)
    parser.add_argument('--use_dataset_action', action='store_true', help='use_dataset_action', default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    
    parser.add_argument('--pretrain_timestamp', action='store', type=str, help='pretrain_timestamp, like 2024-03-27_16-52-32', default='', required=False)
    parser.add_argument('--load_config', action='store', type=int, help='load_config', default=1, required=False)
    parser.add_argument('--device', type=str, help='device', default='cuda:0')
    
    # use image
    parser.add_argument('--use_image_front', action='store_true', help='use_front_image',
                        default=False, required=False)
    parser.add_argument('--use_image_left', action='store_true', help='use_left_image',
                        default=False, required=False)
    parser.add_argument('--use_image_right', action='store_true', help='use_right_image',
                        default=False, required=False)
    parser.add_argument('--use_image_high', action='store_true', help='use_high_image',
                        default=False, required=False)
    
    parser.add_argument('--use_depth_front', action='store_true', help='use_depth_front',
                        default=False, required=False)
    parser.add_argument('--use_depth_left', action='store_true', help='use_depth_left',
                        default=False, required=False)
    parser.add_argument('--use_depth_right', action='store_true', help='use_depth_right',
                        default=False, required=False)
    parser.add_argument('--use_depth_high', action='store_true', help='use_depth_high',
                        default=False, required=False)
    
    parser.add_argument('--use_master_left', action='store_true', help='use_master_left',
                        default=False, required=False)
    parser.add_argument('--use_master_right', action='store_true', help='use_master_right',
                        default=False, required=False)
    
    parser.add_argument('--use_puppet_left', action='store_true', help='use_puppet_left',
                        default=False, required=False)
    parser.add_argument('--use_puppet_right', action='store_true', help='use_puppet_right',
                        default=False, required=False)

    # args.arm_steps_length = [x * 2 for x in args.arm_steps_length]
    
    # load IDM
    parser.add_argument("--load_from", type=str, default="output/inference/inference.pt", help="Load IDM from path.")
    # load hdf5
    parser.add_argument("--load_hdf5", type=str, default="", help="Load hdf5 from path.")
    
    # infer with history
    parser.add_argument("--reset_only", action="store_true", default=False, help="Reset the arms.")

    # parser.add_argument("--prompt", action="store", default="move the left arm carefully towards the cup, aligning the gripper with the handle, and gently close the gripper to secure the handle without applying excessive force, while the right arm remains stationary to maintain balance and support if needed.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Use the right arm to grasp the yellow cloth and move it across the table in smooth, steady motions to clean the surface, while ensuring the left arm remains positioned above the table to monitor the process and adjust if necessary to avoid obstacles.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Use the right arm to carefully grasp the red cube, apply gentle pressure to lift it slightly off the surface.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="using left arm, slowly extend the arm toward the basket on the table, targeting the long, brown-tinged baguette and gently grasp it from above, ensuring a secure hold without squashing the bread.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Using right arm, firmly grasp the white dice on the table using the arm, gently flip it until the side with a single dot is facing upwards, ensuring stabilility while rotating.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Using right arm, firmly grasp the white dice on the table using the arm, gently moving it from the right side of the table to the left side, ensuring stabilility while moving.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Using both arms, firmly grasp the white dice on the table using the right arm, then pass it to the left arm, ensuring stabilility while moving.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Using left arm, press down the button of the toaster.", help="Task prompt.")
    # parser.add_argument("--prompt", action="store", default="Using both arms, position the jeans flat on the surface, grasp the waistband and the bottom hem using one hand each, fold the jeans in half by bringing the waistband to meet the bottom hem, ensure the alignment is neat, then fold again by bringing the new waistband edge to meet the bottom hem once more, using both arms to ensure tight folds.", help="Task prompt.")
    parser.add_argument("--prompt", action="store", default="Using both arms, simultaneously position the robot's grippers above the green and red apples located on the desk, adjust the grippers to securely grasp both apples from the top, and then execute a synchronized lift to pick up both apples together.", help="Task prompt.")
    # Extend the right arm towards the cup, positioning the gripper around its handle, while ensuring the left arm remains stationary and balanced, then securely grasp the handle.
    # Use the right arm to carefully grasp the dice, apply gentle pressure to lift it slightly off the surface.

    # collect data during inference
    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect data during inference.")
    parser.add_argument("--generate_task", action="store_true", default=False, help="Automatically generate task.")

    parser.add_argument("--dinov2_name", type=str, default="facebook/dinov2-base", help="dinov2_name.")
    parser.add_argument("--model_name", type=str, default="direction_aware", help="Model name")
    parser.add_argument("--use_transform", action="store_true", default=False, help="Use transform")

    args = parser.parse_args()

    return args
