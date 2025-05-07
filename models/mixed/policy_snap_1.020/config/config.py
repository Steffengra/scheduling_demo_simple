
import logging
from pathlib import (
    Path,
)
from sys import (
    stdout,
)
from numpy import (
    ceil,
)
from numpy.random import (
    default_rng,
)
import tensorflow as tf

from src.data.user import (
    UserNormal,
    UserAmbulance,
)

# vergleich verschiedener ziele? maxminfair, max throughput, prio


class Config:
    def __init__(
            self,
    ) -> None:

        # GENERAL-------------------------------------------------------------------------------------------------------
        self._logging_level_stdio = logging.INFO  # DEBUG < INFO < WARNING < ERROR < CRITICAL
        self._logging_level_file = logging.WARNING
        self._logging_level_tensorflow = logging.INFO

        # SCHEDULING SIM PARAMETERS-------------------------------------------------------------------------------------
        self.num_episodes: int = 60
        self.num_steps_per_episode: int = 50_000

        self.snr_ue_linear: float = 1
        self.num_total_resource_slots: int = 10
        self.num_users: dict = {
            UserNormal: 3,
            UserAmbulance: 1,
        }
        self.max_job_size_resource_slots: dict = {
            'Normal': 7,
            'Ambulance': 5,
        }
        self.probs_new_job: dict = {
            'Normal': 1.0,
            'Ambulance': 0.5,
        }
        self.rayleigh_fading_scale: float = 1e-8

        self.reward_weightings = {
            'sum rate': 1/40,
            'priority missed': -1,
            'fairness': 0.5,
        }

        # LEARNING PARAMETERS-------------------------------------------------------------------------------------------
        self.exploration_noise_decay_start_percent: float = 0.0  # when to start decay in %
        self.exploration_noise_decay_threshold_percent: float = 0.8  # when to decay to 0 in %
        self.exploration_noise_momentum_initial: float = 1.0

        self.experience_buffer_args: dict = {
            'buffer_size': 10_000,  # Num of samples held, FIFO
            'priority_scale_alpha': 0.0,  # alpha in [0, 1], alpha=0 uniform sampling, 1 is fully prioritized sampling
            'importance_sampling_correction_beta': 1.0  # beta in [0%, 100%], beta=100% is full correction
        }
        self.network_args: dict = {
            'value_network_args': {
                'hidden_layer_units': [512, 512, 512],
                'activation_hidden': 'tanh',  # >relu, tanh
                'kernel_initializer_hidden': 'glorot_uniform'  # >glorot_uniform, he_uniform
            },
            'value_network_optimizer': tf.keras.optimizers.Adam,
            'value_network_optimizer_args': {
                'learning_rate': 1e-4,
                # 'learning_rate': PiecewiseConstantDecay([int(0.8*self.num_episodes*200)], [1e-4, 1e-5]),
                'amsgrad': False,
            },

            'policy_network_args': {
                'hidden_layer_units': [512, 512, 512],
                'activation_hidden': 'tanh',  # >relu, tanh
                'kernel_initializer_hidden': 'glorot_uniform'  # >glorot_uniform, he_uniform
            },
            'policy_network_optimizer': tf.keras.optimizers.Adam,
            'policy_network_optimizer_args': {
                'learning_rate': 1e-6,
                # 'learning_rate': PiecewiseConstantDecay([int(0.8*self.num_episodes*200)], [1e-4, 1e-5]),
                'amsgrad': False,
            },

        }
        self.training_args: dict = {
            'training_minimum_experiences': 1_000,  # Min experiences collected before any training steps
            'training_batch_size': 256,  # Num of experiences sampled in one training step
            'training_target_update_momentum_tau': 1e-2,  # How much of the primary network copy to target networks
            'future_reward_discount_gamma': 0.0,  # Exponential future reward discount for stability
        }
        self.training_args_soft_actor_critic: dict = {
            'entropy_scale_alpha_initial': 1.0,  # Weights the 'soft' entropy penalty against the td error
            'target_entropy': 1.0,  # SAC heuristic impl. = product of action_space.shape
            'entropy_scale_optimizer': tf.keras.optimizers.SGD,
            'entropy_scale_optimizer_args': {
                'learning_rate': 1e-4,  # LR=0.0 -> No adaptive entropy scale -> manually tune initial entropy scale
            }
        }

        self._post_init()

    def _post_init(
            self,
    ) -> None:

        # Paths
        self.project_root_path = Path(__file__).parent.parent.parent
        self.models_path = Path(self.project_root_path, 'models')

        # rng
        self.rng = default_rng(seed=None)

        # Logging
        #   get new sub loggers via logger.getChild(__name__) to improve messaging
        self.logger = logging.getLogger()

        self.logfile_path = Path(self.project_root_path, 'outputs', 'logs', 'log.txt')
        self.__logging_setup()

        # Collected args
        self.size_state: int = 3 * sum(self.num_users.values())

        self.soft_actor_critic_args: dict = {
            'rng': self.rng,
            'parent_logger': self.logger,
            **self.training_args,
            **self.training_args_soft_actor_critic,
            'experience_buffer_args': {'rng': self.rng, **self.experience_buffer_args},
            'network_args': {**self.network_args, 'size_state': self.size_state,
                             'num_actions': sum(self.num_users.values())},
        }
        self.td3_actor_critic_args: dict = {
            'rng': self.rng,
            'parent_logger': self.logger,
            **self.training_args,
            'experience_buffer_args': {'rng': self.rng, **self.experience_buffer_args},
            'network_args': {**self.network_args, 'size_state': self.size_state,
                             'num_actions': sum(self.num_users.values())},
        }

        # Arithmetic
        self.steps_total = self.num_episodes * self.num_steps_per_episode
        self.exploration_noise_step_start_decay: int = ceil(
            self.exploration_noise_decay_start_percent * self.num_episodes * self.num_steps_per_episode
        )
        self.exploration_noise_linear_decay_per_step: float = (
            self.exploration_noise_momentum_initial / (
                self.exploration_noise_decay_threshold_percent * (
                    self.num_episodes * self.num_steps_per_episode - self.exploration_noise_step_start_decay
                )
            )
        )

    def __logging_setup(
            self,
    ) -> None:
        logging_formatter = logging.Formatter(
            '{asctime} : {levelname:8s} : {name:30} : {funcName:20s} :: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{',
        )

        # Create Handlers
        logging_file_handler = logging.FileHandler(self.logfile_path)
        logging_stdio_handler = logging.StreamHandler(stdout)

        # Set Logging Level
        logging_file_handler.setLevel(self._logging_level_file)
        logging_stdio_handler.setLevel(self._logging_level_stdio)

        tensorflow_logger = tf.get_logger()
        tensorflow_logger.setLevel(self._logging_level_tensorflow)

        self.logger.setLevel(logging.NOTSET)  # set primary logger level to lowest to catch all

        # Set Formatting
        logging_file_handler.setFormatter(logging_formatter)
        logging_stdio_handler.setFormatter(logging_formatter)

        # Add Handlers
        self.logger.addHandler(logging_file_handler)
        self.logger.addHandler(logging_stdio_handler)

        # Check Log File Size
        large_log_file_size = 30_000_000
        if self.logfile_path.stat().st_size > large_log_file_size:
            self.logger.warning(f'log file size >{large_log_file_size/1_000_000} MB')
