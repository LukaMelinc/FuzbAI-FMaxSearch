from FuzbAISim import FuzbAISim
from log_utils import setup_logging

setup_logging()

sim = FuzbAISim(
    render_gui=True,
    self_play_config={
        "participant_1_spec": "ppo",
        "participant_2_spec": "ppo",
        "participant_1_kwargs": {
            "model_save_path": "trained_models/#26A.pth",
            "load_model": True,
            "inference": True,
            "training_enabeled": False,
        },
        "participant_2_kwargs": {
            "model_save_path": "trained_models/#26B_MID.pth",
            "load_model": True,
            "inference": True,
            "training_enabeled": False,
        },
    },
)

sim.run()