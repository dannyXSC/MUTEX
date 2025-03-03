import h5py
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from tools.rotation_embed import EulerRotationEmbed
import pickle


def save_video_to_img(vid, path, sample_rate=1):
    length = len(vid)
    cnt = 0
    for i in range(0, length, sample_rate):
        img = Image.fromarray(vid[i])
        img.save(os.path.join(path, f"img_{cnt:03d}.png"))
        cnt += 1


def pre_process(s_end_state, mean, std):
    rotation_embed = EulerRotationEmbed()
    # 创建一个shape除了最后一维为10其他与s_end_state相同的np数组
    new_shape = s_end_state.shape[:-1] + (9,)
    result = np.zeros(new_shape)  # 可以用 np.zeros 或 np.empty，根据需求决定

    result[..., :3] = (s_end_state[..., :3] - mean[:3]) / std[:3]
    # angle
    result[..., 3:9] = rotation_embed.angles_to_embed(s_end_state[..., 3:6])
    return result


def pre_process_action(s_end_state, mean, std):
    rotation_embed = EulerRotationEmbed()
    # 创建一个shape除了最后一维为10其他与s_end_state相同的np数组
    new_shape = s_end_state.shape[:-1] + (10,)
    result = np.zeros(new_shape)  # 可以用 np.zeros 或 np.empty，根据需求决定

    result[..., :3] = (s_end_state[..., :3] - mean[:3]) / std[:3]
    # angle
    result[..., 3:9] = rotation_embed.angles_to_embed(s_end_state[..., 3:6])
    result[..., 9] = s_end_state[..., 6]
    return result


n_ts_per_task = 11

RAW_DATASET_DISCRIPTION = {
    "grab_cube2_v1": "grab the green cube into the plate",
    "grab_cup_v1": "grab the cup and change its position",
    "grab_pencil1_v1": "grab the black pen into the plate",
    "grab_pencil2_v1": "grab the red pen into the plate",
    "grab_to_plate1_and_back_v1": "grab the red cube into the green plate",
    "grab_to_plate1_v1": "grab the red cube into the green plate",
    "grab_to_plate2_and_back_v1": "grab the red cube into the yellow plate",
    "grab_to_plate2_v1": "grab the red cube into the yellow plate",
    "grab_to_plate2_and_pull_v1": "grab the red cube into the green plate and pull the plate",
    "grab_two_cubes2_v1": "grab the green cube into the plate",
    "pull_plate_v1": "pull the plate",
    "push_box_common_v1": "push the box",
    "push_box_random_v1": "push the box",
    "push_box_two_v1": "push the box",
    "push_plate_v1": "push the plate",
}

base_source_folder = "/data1/dataset"
base_dest_folder = "/home/ubuntu/danny/MUTEX/dataset/rw_h2r"
stats_path = "./tools/stats.pkl"

# for normalization
with open(stats_path, "rb") as f:
    stats = pickle.load(f)

for task_name in RAW_DATASET_DISCRIPTION.keys():
    source_folder = os.path.join(base_source_folder, task_name)
    # get h5py file under source folder
    source_file_list = [f for f in os.listdir(source_folder) if f.endswith(".hdf5")]
    # create if not exists
    if not os.path.exists(base_dest_folder):
        os.makedirs(base_dest_folder)
    # get source file name
    # file_name = f"RW_{'_'.join(RAW_DATASET_DISCRIPTION[task_name].split(' '))}_demo"
    file_name = f"RW_{task_name}_demo"
    destination_file = os.path.join(
        base_dest_folder,
        f"{file_name}.hdf5",
    )

    cnt = 0
    with h5py.File(destination_file, "w") as dest_file:
        data = dest_file.create_group("data")
        for source_file_name in tqdm(
            source_file_list, desc=f"Converting task: {task_name}"
        ):
            demo = data.create_group(f"demo_{cnt}")
            obs = demo.create_group("obs")
            source_file = os.path.join(source_folder, source_file_name)
            with h5py.File(source_file, "r") as src_file:
                human_video = src_file["/cam_data/human_camera"][()].astype(np.uint8)
                robot_video = src_file["/cam_data/robot_camera"][()].astype(np.uint8)
                action = pre_process_action(
                    src_file["/action"][()], stats["action_mean"], stats["action_std"]
                )
                end_position = pre_process(
                    src_file["/end_position"][()],
                    stats["end_state_mean"],
                    stats["end_state_std"],
                )
                gripper_state = src_file["/gripper_state"][()]
                gripper_state = gripper_state[:, np.newaxis]
                # state = np.concatenate((end_position, gripper_state[:, np.newaxis]), axis=1)
                time_stamp = src_file["/timestamp"][()]

                if human_video.shape[3] == 3:
                    human_video = human_video[:, :, :, ::-1]
                if robot_video.shape[3] == 3:
                    robot_video = robot_video[:, :, :, ::-1]

                # 降低频率
                robot_video = robot_video[::4]
                gripper_state = gripper_state[::4]
                end_position = end_position[::4]
                action = action[::4]

                # obs.create_dataset("human_video", data=human_video)
                obs.create_dataset("agentview_rgb", data=robot_video)
                obs.create_dataset("gripper_states", data=gripper_state)
                obs.create_dataset("joint_states", data=end_position)
                demo.create_dataset("actions", data=action)
                demo.attrs["num_samples"] = len(action)

                # save to file...
                if cnt < n_ts_per_task:
                    img_base_path = os.path.join(
                        base_dest_folder, "task_spec", file_name, f"vid_{cnt:03d}"
                    )
                    os.makedirs(img_base_path, exist_ok=True)
                    save_video_to_img(robot_video, img_base_path, sample_rate=1)

            cnt += 1
