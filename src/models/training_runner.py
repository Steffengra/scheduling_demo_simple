
import matplotlib.pyplot as plt
from numpy import (
    ndarray,
    infty,
    ones,
    mean,
)
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from shutil import (
    copytree,
    rmtree,
)

from src.config.config import (
    Config,
)
from src.data.scheduling_data import (
    SchedulingData,
)
from src.models.td3 import (
    TD3ActorCritic,
)


class TrainingRunner:
    def __init__(
            self,
    ) -> None:

        self.config = Config()
        self.rng = self.config.rng

    def add_random_distribution(
            self,
            action: ndarray,  # turns out its much faster to numpy the tensor and then do operations on ndarray
            tau_momentum: float,  # tau * random_distribution + (1 - tau) * action
    ) -> ndarray:
        """
        Mix an action vector with a random_uniform vector of same length
        by tau * random_distribution + (1 - tau) * action
        """
        if tau_momentum == 0.0:
            return action

        # create random action
        random_distribution = self.rng.random(size=sum(self.config.num_users.values()), dtype='float32')
        random_distribution = random_distribution / sum(random_distribution)

        # combine
        noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

        # normalize
        sum_noisy_action = sum(noisy_action)
        if sum_noisy_action != 0:
            noisy_action = noisy_action / sum_noisy_action

        return noisy_action

    def train(
            self,
            training_name: str,
    ) -> None:
        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            recent_reward = mean(episode_metrics['rewards'][max(step_id-100, 0):step_id-1])
            # print(episode_metrics['rewards'][step_id])

            print(f'\rSimulation completed: {progress:.2%}, '
                  f'est. finish {finish_time.hour:02d}:{finish_time.minute:02d}:{finish_time.second:02d}'
                  f', recent reward: {recent_reward:.2f}', end='')

        def anneal_parameters() -> tuple:
            if simulation_step > self.config.exploration_noise_step_start_decay:
                exploration_noise_momentum_new = max(
                    0.0,
                    exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                )
            else:
                exploration_noise_momentum_new = exploration_noise_momentum

            return exploration_noise_momentum_new

        def save_model_checkpoint(extra=None):

            name = f'policy_snap_{extra:.3f}'
            checkpoint_path = Path(
                self.config.models_path,
                training_name,
                name,
            )

            allocator.networks['policy'][0]['primary'].save(checkpoint_path)

            # save config
            copytree(Path(self.config.project_root_path, 'src', '../config'),
                     Path(checkpoint_path, '../config'),
                     dirs_exist_ok=True)

            # clean model checkpoints
            for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
                if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:

                    name = f'policy_snap_{high_score_prior:.3f}'

                    prior_checkpoint_path = Path(
                        self.config.models_path,
                        training_name,
                        name,
                    )
                    rmtree(path=prior_checkpoint_path, ignore_errors=True)
                    high_scores.remove(high_score_prior)

            return checkpoint_path

        training_name = training_name
        real_time_start = datetime.now()

        sim = SchedulingData(config=self.config)
        allocator = TD3ActorCritic(**self.config.td3_actor_critic_args)

        exploration_noise_momentum = self.config.exploration_noise_momentum_initial

        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            # 'value_loss_mean': +infty * ones(self.config.num_episodes),
            # 'priority_timeouts_per_occurrence': +infty * ones(self.config.num_episodes),
        }
        high_score = -infty
        high_scores = []

        for episode_id in range(self.config.num_episodes):

            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                # 'value_losses': +infty * ones(self.config.num_steps_per_episode),
                # 'priority_timeouts': +infty * ones(self.config.num_steps_per_episode),
            }

            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next: ndarray = sim.get_state()

            for step_id in range(self.config.num_steps_per_episode):

                simulation_step = episode_id * self.config.num_steps_per_episode + step_id
                # determine state
                state_current = state_next
                step_experience['state'] = state_current

                # find allocation action based on state
                bandwidth_allocation_solution = allocator.get_action(state_current)
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                    action=bandwidth_allocation_solution,
                    tau_momentum=exploration_noise_momentum
                )
                step_experience['action'] = noisy_bandwidth_allocation_solution

                # step simulation based on action
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                )
                step_experience['reward'] = step_reward

                # determine new state
                state_next = sim.get_state()
                step_experience['next_state'] = state_next

                # save tuple (S, A, r, S_{new})
                allocator.add_experience(experience=step_experience)

                # train allocator off-policy
                allocator.train()

                # anneal parameters
                exploration_noise_momentum = anneal_parameters()

                # log step results
                episode_metrics['rewards'][step_id] = step_experience['reward']

                if step_id % 50 == 0:
                    progress_print()

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
            )
            print('\n')

            if per_episode_metrics['reward_per_step'][episode_id] > high_score:
                high_score = per_episode_metrics['reward_per_step'][episode_id]
                high_scores.append(high_score)
                save_model_checkpoint(high_score)

        fig, ax = plt.subplots()
        sliding_window_average_rewards: ndarray = -infty * ones(len(episode_metrics['rewards']))
        sliding_window_average_rewards[0] = episode_metrics['rewards'][0]
        for value_id in range(1, len(episode_metrics['rewards'])):
            lookback = max(0, value_id-100)
            sliding_window_average_rewards[value_id] = mean(episode_metrics['rewards'][lookback:value_id])
        ax.scatter(range(self.config.num_steps_per_episode), sliding_window_average_rewards)
        plt.show()


if __name__ == '__main__':

    r = TrainingRunner()
    r.train(training_name='test')
