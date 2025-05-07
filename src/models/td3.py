
from numpy import (
    ndarray,
    newaxis,
    concatenate,
)
from numpy.random import (
    default_rng,
)
from tensorflow import (
    math as tf_math,
    linalg as tf_linalg,
    random as tf_random,
    function as tf_function,
    constant as tf_constant,
    float32 as tf_float32,
    squeeze as tf_squeeze,
    concat as tf_concat,
    minimum as tf_minimum,
    clip_by_value as tf_clip_by_value,
    reduce_mean as tf_reduce_mean,
    GradientTape as tf_GradientTape,
)
from keras.models import (
    load_model,
)
from logging import (
    Logger,
)
from pathlib import (
    Path,
)

from src.models.experience_buffer import (
    ExperienceBuffer,
)
from src.models.dqn import (
    ValueNetwork,
    PolicyNetwork,
)


class TD3ActorCritic:

    def __init__(
            self,
            rng: default_rng,
            parent_logger: Logger,
            future_reward_discount_gamma: float,
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
            # TODO: The number of networks should probably be part of config
            for _ in range(2):
                self.networks['value'].append(
                    {
                        'primary': ValueNetwork(**value_network_args),
                        'target': ValueNetwork(**value_network_args),
                    }
                )

            for _ in range(1):
                self.networks['policy'].append(
                    {
                        'primary': PolicyNetwork(num_actions=num_actions, **policy_network_args),
                        'target': PolicyNetwork(num_actions=num_actions, **policy_network_args),
                    }
                )

            # COMPILE NETWORKS
            dummy_state = self.rng.random(size_state)
            dummy_action = self.rng.random(num_actions)

            for network_type, network_list in self.networks.items():
                # create dummy input
                if network_type == 'policy':
                    dummy_input = dummy_state[newaxis]
                    optimizer = policy_network_optimizer
                    optimizer_args = policy_network_optimizer_args
                elif network_type == 'value':
                    dummy_input = concatenate([dummy_state, dummy_action])[newaxis]
                    optimizer = value_network_optimizer
                    optimizer_args = value_network_optimizer_args
                # feed dummy input, compile primary
                for network_pair in network_list:
                    for network_rank, network in network_pair.items():
                        network.initialize_inputs(dummy_input)
                    network_pair['primary'].compile(optimizer=optimizer(**optimizer_args))
            self.update_target_networks(tau_target_update=1.0)

        self.rng: default_rng = rng
        self.logger: Logger = parent_logger.getChild(__name__)

        self.training_minimum_experiences: int = training_minimum_experiences
        self.training_batch_size: int = training_batch_size
        self.future_reward_discount_gamma: float = future_reward_discount_gamma
        self.training_target_update_momentum_tau: float = training_target_update_momentum_tau

        self.experience_buffer = ExperienceBuffer(**experience_buffer_args)

        self.networks: dict = {'value': [], 'policy': []}
        initialize_networks(**network_args)

        self.logger.info('TD3 initialized')

    @tf_function
    def update_target_networks(
            self,
            tau_target_update: float
    ) -> None:

        for network_list in self.networks.values():
            for network_pair in network_list:
                for v_primary, v_target in zip(network_pair['primary'].trainable_variables,
                                               network_pair['target'].trainable_variables):
                    v_target.assign(tau_target_update * v_primary + (1 - tau_target_update) * v_target)

    def save_networks(
            self,
            model_path: Path,
    ) -> None:

        for network_type, network_list in self.networks.items():
            for network_pair_id in range(len(network_list)):
                network_list[network_pair_id]['target'].save(Path(model_path, f'{network_type}_{network_pair_id}'))

    def load_policy(
            self,
            model_path: Path,
    ) -> None:
        # TODO: You want to figure out in which way to load these here, e.g., full path as input?
        for network_pair in self.networks['policy']:
            network_pair['primary'] = load_model(Path(model_path, 'policy'))
            network_pair['target'] = load_model(Path(model_path, 'policy'))
            self.logger.error('this isnt fully implemented, stop')
            exit()

    def get_action(
            self,
            state,
    ) -> ndarray:

        if state.ndim == 1:
            state = state[newaxis]
        action = self.networks['policy'][0]['primary'].call(state)
        return action.numpy().flatten()

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        self.experience_buffer.add_experience(experience=experience)

    @tf_function
    def train_graph(
            self,
            states,
            actions,
            rewards,
            next_states,
            sample_importance_weights,
    ):
        # TODO: This is hardcoded for two value nets because tf minimum function is weird

        # TRAIN VALUE NETWORKS
        target_q = rewards
        # get future reward estimate
        if self.future_reward_discount_gamma > 0:
            next_actions = self.networks['policy'][0]['target'].call(next_states)
            # Add a small amount of random noise to action for smoothing
            # TODO: Reimplement this, maybe
            # noise = tf_random.normal(shape=next_actions.shape,
            #                          mean=0, stddev=training_noise_std, dtype=tf_float32)
            # noise = tf_clip_by_value(noise, -training_noise_clip, training_noise_clip)
            # next_actions += noise
            # next_actions = tf_linalg.normalize(next_actions, axis=1, ord=1)[0]  # re-normalize
            # Clipping so that the extra net won't introduce more overestimation
            input_vector = tf_concat([next_states, next_actions], axis=1)
            q_estimate_1 = self.networks['value'][0]['target'].call(input_vector)
            q_estimate_2 = self.networks['value'][1]['target'].call(input_vector)
            conservative_q_estimate = tf_squeeze(tf_minimum(q_estimate_1, q_estimate_2))
            target_q = target_q + self.future_reward_discount_gamma * conservative_q_estimate

        input_vector = tf_concat([states, actions], axis=1)
        for network_pair in self.networks['value']:
            with tf_GradientTape() as tape:  # autograd
                estimate = tf_squeeze(network_pair['primary'].call(input_vector))
                td_error = tf_math.subtract(target_q, estimate)
                loss_estimation = tf_reduce_mean(
                    sample_importance_weights * td_error ** 2
                )
                loss = (
                    loss_estimation
                )

            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=network_pair['primary'].trainable_variables)
            network_pair['primary'].optimizer.apply_gradients(  # apply gradient update
                zip(gradients, network_pair['primary'].trainable_variables))

        # TRAIN POLICY NETWORKS
        input_vector = states
        for network_pair in self.networks['policy']:
            with tf_GradientTape() as tape:  # autograd
                actor_actions = network_pair['primary'].call(input_vector)
                value_network_input = tf_concat([input_vector, actor_actions], axis=1)
                # Original Paper, DDPG Paper and other implementations train on primary network. Why?
                #  Because otherwise the value net is always one gradient step behind
                # TODO: This always uses the first value network
                value_network_score = tf_reduce_mean(
                    self.networks['value'][0]['primary'].call(value_network_input)
                )
                loss = (
                    - value_network_score
                )

            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=network_pair['primary'].trainable_variables)
            network_pair['primary'].optimizer.apply_gradients(  # apply gradient update
                zip(gradients, network_pair['primary'].trainable_variables))

        self.update_target_networks(tau_target_update=self.training_target_update_momentum_tau)
        # TODO: i think we're not adjusting the priorities anywhere for prio exp replay

    def train(
            self,
    ) -> None:

        if (self.experience_buffer.get_len() < self.training_minimum_experiences) or (
                self.experience_buffer.get_len() < self.training_batch_size):
            return

        # SAMPLE FROM BUFFER
        (
            sample_experiences,
            sample_experience_ids,
            sample_importance_weights,
        ) = self.experience_buffer.sample(batch_size=self.training_batch_size)

        states = tf_constant([experience['state'] for experience in sample_experiences], dtype=tf_float32)
        actions = tf_constant([experience['action'] for experience in sample_experiences], dtype=tf_float32)
        rewards = tf_constant([experience['reward'] for experience in sample_experiences], dtype=tf_float32)
        next_states = tf_constant([experience['next_state'] for experience in sample_experiences], dtype=tf_float32)
        sample_importance_weights = tf_constant(sample_importance_weights, dtype=tf_float32)

        self.train_graph(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            sample_importance_weights=sample_importance_weights,
        )


