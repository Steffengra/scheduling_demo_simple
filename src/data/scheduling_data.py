
from numpy import (
    ndarray,
    zeros,
    multiply,
    round as np_round,
    minimum,
    log2,
    mean,
    std,
)
from copy import (
    deepcopy,
)

from src.data.resource_grid import (
    ResourceGrid,
)


class SchedulingData:
    def __init__(
            self,
            config,
    ) -> None:

        self.config = config
        self.logger = self.config.logger.getChild(__name__)
        self.rng = self.config.rng

        # INITIALIZE RESOURCE GRID
        self.resource_grid = ResourceGrid(total_resource_slots=self.config.num_total_resource_slots)
        self.logger.info('ResourceGrid initialized')

        # INITIALIZE USERS
        self.users = {}
        user_id = 0
        for user_type, user_type_amount in self.config.num_users.items():
            for user_type_idx in range(user_type_amount):
                self.users[user_id] = (
                    user_type(
                        user_id=user_id,
                        max_job_sizes_resource_slots=self.config.max_job_size_resource_slots,
                        rayleigh_fading_scale=self.config.rayleigh_fading_scale,
                        probs_new_job=self.config.probs_new_job,
                        rng=self.rng,
                        parent_logger=self.logger
                    )
                )
                user_id += 1
        self.logger.info('Users initialized')

        self.generate_new_jobs()
        self.logger.info('SchedulingData sim initialized')

    def export_state(
            self,
    ) -> dict:
        state: dict = {
            'resource_grid': deepcopy(self.resource_grid),
            'users': deepcopy(self.users)
        }
        return state

    def import_state(
            self,
            state: dict,
    ) -> None:
        self.resource_grid = state['resource_grid']
        self.users = state['users']

    def generate_new_jobs(
            self,
    ) -> None:

        for user in self.users.values():
            user.generate_job()

    def update_user_power_gain(
            self,
    ) -> None:

        for user in self.users.values():
            user.update_power_gain()

    def get_state(
            self,
    ) -> ndarray:

        # per user: channel conditions, packets, priority
        state_length = 3 * len(self.users)
        state = zeros(state_length, dtype='float32')

        state[0:len(self.users)] = [user.power_gain for user in self.users.values()]
        state[len(self.users):2*len(self.users)] = [user.job.size_resource_slots if user.job else 0
                                                    for user in self.users.values()]
        state[2*len(self.users):3*len(self.users)] = [user.job.priority if user.job else 0
                                                      for user in self.users.values()]

        self.logger.debug(f'Current state is {state}')

        return state.astype('float32')

    def step(
            self,
            percentage_allocation_solution: ndarray,
    ) -> tuple[dict, dict]:

        # Convert percentage allocation into slot allocation, but at most as many res as requested
        requested_slots_per_ue = [
            self.users[ue_id].job.size_resource_slots if self.users[ue_id].job else 0
            for ue_id in range(len(self.users))
        ]

        slot_allocation_solution = [
            minimum(
                np_round(percentage_allocation_solution[ue_id] * self.resource_grid.total_resource_slots),
                requested_slots_per_ue[ue_id],
                dtype='float32'
            )
            for ue_id in range(len(self.users))
        ]

        # grant at most one additional resource if there was rounding down
        if sum(slot_allocation_solution) == self.resource_grid.total_resource_slots - 1:
            remainders = np_round([
                percentage_allocation_solution[ue_id] * self.resource_grid.total_resource_slots - slot_allocation_solution[ue_id]
                for ue_id in range(len(self.users))
            ], decimals=5)
            for ue_id in range(len(self.users)):
                if remainders[ue_id] > 0:
                    if requested_slots_per_ue[ue_id] > slot_allocation_solution[ue_id]:
                        slot_allocation_solution[ue_id] += 1
                        break

        # Check if the rounding has resulted in more resources distributed than available
        if sum(slot_allocation_solution) > self.resource_grid.total_resource_slots:
            # if so, remove one resource from a random user
            while sum(slot_allocation_solution) > self.resource_grid.total_resource_slots:
                random_user_id = self.rng.integers(0, len(self.users))
                if slot_allocation_solution[random_user_id] > 0:
                    slot_allocation_solution[random_user_id] -= 1

        # prepare the allocated slots per ue for metrics calculation
        allocated_slots_per_ue: dict = {
                ue_id: slot_allocation_solution[ue_id]
                for ue_id in range(len(self.users))
        }

        self.logger.debug(f'allocated slots per ue: {allocated_slots_per_ue}')
        if sum(allocated_slots_per_ue.values()) > self.resource_grid.total_resource_slots:
            self.logger.error('ALAAARM too many resources allocated')
            exit()

        # calculate sum rate
        sum_rate_capacity_bit_per_second: float = 0.0
        for user_id, allocated_slots in allocated_slots_per_ue.items():
            sum_rate_capacity_bit_per_second += (
                allocated_slots * log2(1 + self.users[user_id].power_gain * self.config.snr_ue_linear)
            )

        self.logger.debug(f'sum rate {sum_rate_capacity_bit_per_second}')

        # see how many priority==1 jobs were not fully transmitted
        priority_jobs_missed_counter: int = 0
        for user_id in allocated_slots_per_ue.keys():
            if self.users[user_id].job:
                if self.users[user_id].job.priority == 1:
                    self.logger.debug(f'Priority job requested {self.users[user_id].job.size_resource_slots} received {allocated_slots_per_ue[user_id]}')
                    if allocated_slots_per_ue[user_id] < self.users[user_id].job.size_resource_slots:
                        priority_jobs_missed_counter += 1

        self.logger.debug(f'prio jobs missed {priority_jobs_missed_counter}')

        # calculate jain's fairness score
        #  result ranges from 1/n (worst) to 1.0 (best)
        power_gains = [user.power_gain for user in self.users.values()]
        weighted_slots_per_ue = multiply(list(allocated_slots_per_ue.values()), power_gains)
        #  middle out jobs that requested little, so they don't ruin fairness even though they didn't want more
        for ue_id in range(len(weighted_slots_per_ue)):
            if requested_slots_per_ue[ue_id] <= allocated_slots_per_ue[ue_id]:
                weighted_slots_per_ue[ue_id] = mean(weighted_slots_per_ue)

        if sum(weighted_slots_per_ue) > 0:
            fairness_score = 1 / (
                    1 + (std(weighted_slots_per_ue) / mean(weighted_slots_per_ue))**2
            )
            fairness_score = fairness_score.astype('float32')
        else:
            if sum(requested_slots_per_ue) == 0:
                fairness_score = 1.0
            else:
                fairness_score = 0.0

        # transform fairness score to [0.. 1]?
        fairness_score = (fairness_score - 1/len(self.users)) / (1 - 1/len(self.users))

        self.logger.debug(f'channel weighted slots per ue {weighted_slots_per_ue}')
        self.logger.debug(f'fairness_score {fairness_score}')

        # prepare reward metric
        reward = (
            + self.config.reward_weightings['sum rate'] * sum_rate_capacity_bit_per_second
            + self.config.reward_weightings['priority missed'] * priority_jobs_missed_counter
            + self.config.reward_weightings['fairness'] * fairness_score
        ).astype('float32')

        reward_components = {
            'sum rate': sum_rate_capacity_bit_per_second,
            'prio jobs missed': priority_jobs_missed_counter,
            'weighted slots per ue': weighted_slots_per_ue.astype('float32'),
            'fairness score': fairness_score,
        }

        # move sim to new state
        self.update_user_power_gain()
        self.generate_new_jobs()

        return reward, reward_components
