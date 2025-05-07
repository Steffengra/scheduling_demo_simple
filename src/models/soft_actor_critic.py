
from logging import (
    Logger,
)
from numpy import (
    ndarray,
    concatenate as np_concatenate,
    newaxis as np_newaxis,
    log as np_log,
)
from numpy.random import (
    default_rng,
)
from tensorflow import (
    Variable as tf_Variable,
    float32 as tf_float32,
    function as tf_function,
)
from keras.models import (
    load_model,
)
from pathlib import (
    Path,
)

from src.models.experience_buffer import (
    ExperienceBuffer,
)
from src.models.dqn import (
    ValueNetwork,
    PolicyNetworkSoft,
)


class SoftActorCritic:
    def __init__(
            self,
            rng: default_rng,
            parent_logger: Logger,
            future_reward_discount_gamma: float,
            entropy_scale_alpha_initial: float,
            target_entropy: float,
            entropy_scale_optimizer,
            entropy_scale_optimizer_args: dict,
            training_minimum_experiences: int,
            training_batch_size: int,
            training_target_update_momentum_tau: float,
            experience_buffer_args: dict,
            network_args: dict,
    ) -> None:

        def initialize_networks(
            value_network_args: dict,
            value_network_optimizer: callable,
            value_network_optimizer_args: dict,
            policy_network_args: dict,
            policy_network_optimizer: callable,
            policy_network_optimizer_args: dict,
            size_state: int,
            num_actions: int,
        ) -> None:

            # CREATE NETWORKS
            #  Create target and primary value networks and policy network
            for value_network_id in range(2):
                self.value_networks[value_network_id] = {name: ValueNetwork(**value_network_args)
                                                         for name in ['primary', 'target']}
            self.policy_network = PolicyNetworkSoft(num_actions=num_actions, **policy_network_args)

            # INITIALIZE NETWORKS
            #  Assign optimizer, initialize primary weights, and copy weights to target networks
            dummy_state = self.rng.random(size_state)
            dummy_action = self.rng.random(num_actions)
            for network_id in self.value_networks:
                self.value_networks[network_id]['primary'].compile(
                    optimizer=value_network_optimizer(**value_network_optimizer_args))
                for network in self.value_networks[network_id].values():
                    network(np_concatenate([dummy_state, dummy_action])[np_newaxis])
            self.policy_network.compile(
                optimizer=policy_network_optimizer(**policy_network_optimizer_args))
            self.policy_network(dummy_state[np_newaxis])
            self.update_target_networks(tau_target_update=1.0)

        self.rng = rng
        self.logger = parent_logger.getChild(__name__)

        self.future_reward_discount_gamma = future_reward_discount_gamma

        # Gradients are applied on the log value. This way, entropy_scale_alpha is restricted to positive range
        self.log_entropy_scale_alpha = tf_Variable(np_log(entropy_scale_alpha_initial),
                                                   trainable=True, dtype=tf_float32)
        self.target_entropy = target_entropy
        self.entropy_scale_alpha_optimizer = entropy_scale_optimizer(**entropy_scale_optimizer_args)

        self.training_minimum_experiences = training_minimum_experiences
        self.training_batch_size = training_batch_size
        self.training_target_update_momentum_tau = training_target_update_momentum_tau

        self.experience_buffer = ExperienceBuffer(**experience_buffer_args)

        self.value_networks: dict = {}
        self.policy_network = None
        initialize_networks(**network_args)

        self.logger.info('SoftActorCritic initialized')

    def save_networks(
            self,
            model_path: Path,
    ) -> None:
        self.value_networks[0]['target'].save(Path(model_path, 'value1'))
        self.value_networks[1]['target'].save(Path(model_path, 'value2'))
        self.policy_network.save(Path(model_path, 'policy'))

    def load_policy(
            self,
            model_path: Path,
    ) -> None:
        self.policy_network = load_model(Path(model_path, 'policy'))

    def get_action(
            self,
            state,
    ) -> ndarray:
        _, _, actions_softmax = self.policy_network.get_action_and_log_prob_density(state=state)

        return actions_softmax.numpy().flatten()

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        self.experience_buffer.add_experience(experience=experience)

    @tf_function
    def update_target_networks(
            self,
            tau_target_update: float
    ) -> None:
        # Value networks
        for network_id in self.value_networks:
            for v_primary, v_target in zip(self.value_networks[network_id]['primary'].trainable_variables,
                                           self.value_networks[network_id]['target'].trainable_variables):
                v_target.assign(tau_target_update * v_primary + (1 - tau_target_update) * v_target)
