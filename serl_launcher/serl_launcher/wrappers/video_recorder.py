import os
from typing import List, Optional
import gymnasium as gym
import imageio
import numpy as np
import torch
from pathlib import Path


def compose_frames(
    all_frames: List[np.ndarray],
    num_videos_per_row: int,
    margin: int = 4,
) -> List[np.ndarray]:
    """Compose multiple video frames into a grid layout."""
    num_episodes = len(all_frames)
    num_videos_per_row = num_videos_per_row or num_episodes

    t = 0
    end_of_all_epidoes = False
    frames_to_save = []
    
    while not end_of_all_epidoes:
        frames_t = []

        for i in range(num_episodes):
            # If the episode is shorter, repeat the last frame
            t_ = min(t, len(all_frames[i]) - 1)
            frame_i_t = all_frames[i][t_]

            # Add margins
            frame_i_t = np.pad(
                frame_i_t,
                [[margin, margin], [margin, margin], [0, 0]],
                mode="constant",
                constant_values=0,
            )
            frames_t.append(frame_i_t)

        # Arrange videos in grid
        frame_t = None
        while len(frames_t) >= num_videos_per_row:
            frames_t_this_row = frames_t[:num_videos_per_row]
            frames_t = frames_t[num_videos_per_row:]

            frame_t_this_row = np.concatenate(frames_t_this_row, axis=1)
            if frame_t is None:
                frame_t = frame_t_this_row
            else:
                frame_t = np.concatenate([frame_t, frame_t_this_row], axis=0)

        frames_to_save.append(frame_t)
        t += 1
        end_of_all_epidoes = all([len(all_frames[i]) <= t for i in range(num_episodes)])

    return frames_to_save


class VideoRecorder(gym.Wrapper):
    """Wrapper for recording videos of environment episodes."""
    
    def __init__(
        self,
        env: gym.Env,
        save_folder: str = "",
        save_prefix: Optional[str] = None,
        height: int = 128,
        width: int = 128,
        fps: int = 30,
        camera_id: int = 0,
        goal_conditioned: bool = False,
    ):
        super().__init__(env)
        
        self.save_folder = Path(save_folder)
        self.save_prefix = save_prefix
        self.height = height
        self.width = width
        self.fps = fps
        self.camera_id = camera_id
        self.frames = []
        self.goal_conditioned = goal_conditioned

        # Create save directory if it doesn't exist
        self.save_folder.mkdir(parents=True, exist_ok=True)

        self.num_record_episodes = -1
        self.num_videos = 0
        self.current_save_path = None

    def start_recording(self, num_episodes: Optional[int] = None, num_videos_per_row: Optional[int] = None):
        """Start recording episodes."""
        if num_videos_per_row is not None and num_episodes is not None:
            assert num_episodes >= num_videos_per_row

        self.num_record_episodes = num_episodes
        self.num_videos_per_row = num_videos_per_row
        self.all_frames = []

    def stop_recording(self):
        """Stop recording episodes."""
        self.num_record_episodes = None

    def step(self, action: np.ndarray):
        """Execute environment step and record frame if recording is enabled."""
        if self.num_record_episodes is None or self.num_record_episodes == 0:
            return self.env.step(action)

        elif self.num_record_episodes > 0:
            # Render frame
            frame = self.env.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )

            if frame is None:
                try:
                    frame = self.sim.render(
                        width=self.width,
                        height=self.height,
                        mode="offscreen"
                    )
                    frame = np.flipud(frame)
                except Exception:
                    raise NotImplementedError("Rendering is not implemented.")

            self.frames.append(frame.astype(np.uint8))

            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                if self.goal_conditioned:
                    frames = [
                        np.concatenate([self.env.current_goal["image"], frame], axis=0)
                        for frame in self.frames
                    ]
                else:
                    frames = self.frames

                self.all_frames.append(frames)

                if self.num_record_episodes > 0:
                    self.num_record_episodes -= 1

                if self.num_record_episodes is None:
                    # Save single episode
                    frames_to_save = frames
                    should_save = True
                elif self.num_record_episodes == 0:
                    # Save grid of episodes
                    frames_to_save = compose_frames(
                        self.all_frames,
                        self.num_videos_per_row
                    )
                    should_save = True
                else:
                    should_save = False

                if should_save:
                    filename = f"{self.num_videos:08d}.mp4"
                    if self.save_prefix:
                        filename = f"{self.save_prefix}_{filename}"
                    
                    self.current_save_path = self.save_folder / filename
                    imageio.mimsave(
                        self.current_save_path,
                        frames_to_save,
                        format='MP4',
                        fps=self.fps
                    )
                    self.num_videos += 1

                self.frames = []

            return observation, reward, terminated, truncated, info

        else:
            raise ValueError("Do not forget to call start_recording.")
