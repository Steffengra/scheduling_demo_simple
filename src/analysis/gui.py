
from pathlib import Path
from sys import path as sys_path

from tensorflow.python.ops.parallel_for.pfor import passthrough_stateful_ops

project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import tkhtmlview as tkhtml

import numpy as np

from src.config.config import (
    Config,
)
from src.config.config_gui import (
    ConfigGUI,
)
from src.data.scheduling_data import (
    SchedulingData,
)
from src.analysis.gui_elements import (
    Scenario,
    ScreenSelector,
    ScreenActionButtons,
    ScreenAllocations,
    ScreenLifetimeStats,
)
from src.utils.get_width_rescale_constant_aspect_ratio import get_width_rescale_constant_aspect_ratio


class App(tk.Tk):
    """
    The main GUI object. Populated with elements from gui_elements.py.
    """

    def __init__(
            self,
    ) -> None:
        """
        Create GUI, set params like fullscreen, get some device info like window width in px.
        Also set up simulations for evaluating allocations.
        """

        super().__init__()
        self.configure(bg='white')
        self.attributes('-fullscreen', True)

        # Load config files
        self.config = Config()  # sim related
        self.config_gui = ConfigGUI()  # gui related

        # Get device info
        self.window_width = self.winfo_screenwidth()
        self.window_height = self.winfo_screenheight()
        self.pixels_per_inch = int(self.winfo_fpixels('1i'))

        # Globals for the countdown button
        self.countdown_toggle = False
        self.countdown_value = 0

        # Auto mode toggle
        self.auto_mode_toggle = False

        # tutorial toggle
        self.tutorial_toggle = False

        self._idle_job = None

        # Set up sims to evaluate allocations
        self.rigged_start_states = self.config_gui.rigged_start_states.copy()

        self.sim_main = SchedulingData(config=self.config)  # user sim
        if len(self.rigged_start_states) > 0:
            rigged_state = self.rigged_start_states.pop(0)
            self.rig_sim_main_state(state=rigged_state)

        self.secondary_simulations = {  # learned algorithm sims
            learner_name: SchedulingData(config=self.config)
            for learner_name in self.config_gui.learned_agents.keys()
        }
        self.update_secondary_simulations()  # secondary sims copy the main sim state

        # Global store for user allocation
        self.resources_per_user = {
            user_id: 0
            for user_id in range(4)
        }

        # Lifetime stat keeping
        self.lifetime_stats = {  # user
            'self': {
                'sumrate': [],
                'fairness': [],
                'timeouts': [],
                'overall': [],
            }
        }
        for learner_name in self.config_gui.learned_agents.keys():  # learned algorithms
            self.lifetime_stats[learner_name] = {
                'sumrate': [],
                'fairness': [],
                'timeouts': [],
                'overall': [],
            }

        # Arithmetic
        self.channel_img_height = int(self.config_gui.label_user_font[1]*1.2)
        self.label_base_station_height = int(self.config_gui.label_img_base_station_height_scale * self.window_height)

        # Set up GUI elements
        self._gui_setup()

    def _gui_setup(
            self,
    ) -> None:
        """
        Set up & position GUI elements from gui_elements.py.
        """

        # load tutorial image for later use
        self.image_tutorial = Image.open(Path(project_root_path, 'src', 'analysis', 'img', '00_ANTposter_mini.png'))
        self.tk_image_tutorial = ImageTk.PhotoImage(self.image_tutorial.resize((
            get_width_rescale_constant_aspect_ratio(self.image_tutorial, int(0.9*self.window_height)),
            int(0.9*self.window_height),
        )))

        # Load channel strength indicator images for later use
        self.images_channelstrength = [
            Image.open(Path(project_root_path, 'src', 'analysis', 'img', channel_strength_indicator_img))
            for channel_strength_indicator_img in self.config_gui.channel_strength_indicator_imgs
        ]
        self.tk_images_channel_strength = [
            ImageTk.PhotoImage(image_channel_strength.resize((
                get_width_rescale_constant_aspect_ratio(image_channel_strength, self.channel_img_height),
                self.channel_img_height,
            )))
            for image_channel_strength in self.images_channelstrength
        ]

        # Load button images for later use
        button_timer_image_height = int(self.config_gui.button_countdown_img_scale * self.window_height)
        button_countdown_image = Image.open(Path(project_root_path, 'src', 'analysis', 'img', self.config_gui.button_countdown_img))
        button_countdown_image = button_countdown_image.resize((
            get_width_rescale_constant_aspect_ratio(button_countdown_image, button_timer_image_height),
            button_timer_image_height,
        ))
        self.tk_image_button_countdown = ImageTk.PhotoImage(button_countdown_image)
        self.config_gui.button_countdown_config['image'] = self.tk_image_button_countdown

        button_auto_image = Image.open(Path(project_root_path, 'src', 'analysis', 'img', self.config_gui.button_auto_img))
        button_auto_image = button_auto_image.resize((
            get_width_rescale_constant_aspect_ratio(button_auto_image, button_timer_image_height),
            button_timer_image_height,
        ))
        self.tk_image_button_auto = ImageTk.PhotoImage(button_auto_image)
        self.config_gui.button_auto_config['image'] = self.tk_image_button_auto

        button_reset_image = Image.open(Path(project_root_path, 'src', 'analysis', 'img', self.config_gui.button_reset_img))
        button_reset_image = button_reset_image.resize((
            get_width_rescale_constant_aspect_ratio(button_reset_image, button_timer_image_height),
            button_timer_image_height,
        ))
        self.tk_image_button_reset = ImageTk.PhotoImage(button_reset_image)
        self.config_gui.button_reset_config['image'] = self.tk_image_button_reset

        # tutorial overlay
        # self.frame_tutorial = tk.Frame(
        #     master=self,
        #     width=.3 * self.window_width, height=0.8 * self.window_height,
        #     **self.config_gui.frames_config
        # )

        # Scenario - left hand side of the screen
        self.frame_scenario = Scenario(
            master=self,
            config_gui=self.config_gui,
            window_width=self.window_width,
            window_height=self.window_height,
            logo_paths=[Path(project_root_path, 'src', 'analysis', 'img', logo) for logo in self.config_gui.logos],
            user_image_paths=[Path(project_root_path, 'src', 'analysis', 'img', user_image) for user_image in self.config_gui.user_images],
            base_station_image_path=Path(project_root_path, 'src', 'analysis', 'img', self.config_gui.base_station_image),
            button_callbacks=[
                lambda event: self.allocate_resource(user_id=0),
                lambda event: self.allocate_resource(user_id=1),
                lambda event: self.allocate_resource(user_id=2),
                lambda event: self.allocate_resource(user_id=3),
            ],
            language_button_callbacks=[
                lambda event: self.change_language('DE'),
                lambda event: self.change_language('EN'),
            ],
            tutorial_button_callback=self.callback_button_toggle_tutorial,
            num_total_resource_slots=self.config.num_total_resource_slots,
            **self.config_gui.frames_config,
        )

        # Screen selector buttons - top right of screen
        # self.frame_screen_selector = ScreenSelector(
        #     master=self,
        #     config_gui=self.config_gui,
        #     window_width=self.window_width,
        #     window_height=self.window_height,
        #     button_commands=[self.change_to_frame_allocations, self.change_to_frame_instant_stats, self.change_to_frame_lifetime_stats],
        #     **self.config_gui.frames_config,
        # )

        # Screen action buttons - below screen selector buttons
        # self.frame_screen_action_buttons = ScreenActionButtons(
        #     master=self,
        #     config_gui=self.config_gui,
        #     window_width=self.window_width,
        #     window_height=self.window_height,
        #     button_commands=[self.callback_button_timer, self.callback_button_auto_mode, self.callback_button_reset],
        #     **self.config_gui.frames_config,
        # )



        # Results - bottom right hand side of screen, switched via screen selector buttons
        self.frame_allocations = ScreenAllocations(
            master=self,
            config_gui=self.config_gui,
            window_width=self.window_width,
            window_height=self.window_height,
            pixels_per_inch=self.pixels_per_inch,
            num_total_resource_slots=self.config.num_total_resource_slots,
            callback_next_allocation=self.callback_next_allocation,
            **self.config_gui.frames_config,
        )

        width_explanations = .35 * self.window_width
        wraplength_explanations = .33*self.window_width
        self.frame_explanation = tk.Frame(self, width=width_explanations, height=0.9 * self.window_height, **self.config_gui.frames_config)
        self.label_explanations = tk.Label(self.frame_explanation, wraplength=wraplength_explanations, **self.config_gui.label_explanations_text_config)
        self.label_explanations.config(text=self.config_gui.label_explanation_text1)

        self.frame_results_explanation = tk.Frame(self, width=width_explanations, height=0.9 * self.window_height, **self.config_gui.frames_config)
        self.label_results_explanation = tk.Label(self.frame_results_explanation, wraplength=wraplength_explanations, **self.config_gui.label_explanations_text_config)
        self.button_next_ai = tk.Button(self.frame_results_explanation, command=self.callback_button_next_ai, **self.config_gui.button_next_ai_config)

        # self.label_explanations = tkhtml.HTMLLabel(self.frame_explanation, html='abc', bg='white', fg='white', bd=0)
        # self.label_explanations = tk.Text(self.frame_explanation, relief='flat', bd=0)
        # self.label_explanations.insert(tk.END, 'abc\nabc')

        # Place frames
        self.frame_scenario.place(relx=0.0)

        # self.frame_screen_selector.place(relx=.7, rely=0.1)
        # self.frame_screen_selector.pack_propagate(False)

        # self.frame_screen_action_buttons.place(relx=.7, rely=0.0)
        # self.frame_screen_action_buttons.pack_propagate(False)

        self.label_explanations.pack()
        self.label_results_explanation.pack()
        self.button_next_ai.pack()

        explanations_relx = 0.01
        explanations_rely = 0.23

        self.frame_explanation.place(relx=explanations_relx, rely=explanations_rely)
        self.frame_explanation.pack_propagate(False)

        self.frame_results_explanation.place(relx=explanations_relx, rely=explanations_rely)
        self.frame_results_explanation.pack_propagate(False)

        self.frame_allocations.place(relx=explanations_relx, rely=explanations_rely)
        self.frame_allocations.pack_propagate(False)

        self.frame_explanation.tkraise()




        # self.frame_lifetime_stats.place(relx=.7, rely=0.2)
        # self.frame_lifetime_stats.pack_propagate(False)

        # Separator Vertical
        # self.separator_vertical = ttk.Separator(self, orient='vertical')
        # self.separator_vertical.place(relx=0.7, rely=0, relwidth=0.0, relheight=1)

        # Aggregate button-selectable frames for easier bookkeeping
        # self.selectable_frames = {
        #     'Allocations': self.frame_allocations,
        #     'LifetimeStats': self.frame_lifetime_stats,
        # }

        # Raise as first frame
        # self.change_to_frame_allocations()

        # Initialize user text labels
        self.update_user_text_labels()

        # self.label_tutorial_img = tk.Label(
        #     master=self.frame_tutorial,
        #     image=self.tk_image_tutorial,
        #     bg='white',
        #     highlightthickness=15,
        #     highlightbackground='black',
        # )
        # self.label_tutorial_img.pack()
        # self.frame_tutorial.place(relx=.3, rely=0.04)
        # self.frame_tutorial.tkraise()

    def reset(
            self,
    ) -> None:

        self.callback_next_allocation()

        # reset rigged start states
        self.rigged_start_states = self.config_gui.rigged_start_states.copy()
        if len(self.rigged_start_states) > 0:
            rigged_state = self.rigged_start_states.pop(0)
            self.rig_sim_main_state(state=rigged_state)
            self.update_user_text_labels()
            self.update_secondary_simulations()

    def check_loop(
            self,
    ) -> None:
        """
        Function that is called periodically.
        If countdown mode is active,
            1) If countdown value > 0, decrement countdown value, update countdown value indicator
            2) If countdown value == 0, evaluate the current allocation and reset the timer, update indicator
        Then call self again with a delay.
        """

        if self.countdown_toggle:
            if self.countdown_value == 0:
                self.evaluate_allocation()
                self.countdown_value = self.config_gui.countdown_reset_value_seconds

            self.countdown_value -= 1
            self.frame_screen_action_buttons.button_countdown.configure(text=f'{self.countdown_value}', image=self.tk_image_button_countdown)  # workaround so button doesn't resize on click
            self.after(1000, self.check_loop)

    def change_to_frame_allocations(
            self,
    ) -> None:
        """
        Raise frame results to the top.
        """

        self.frame_allocations.tkraise()
        # self.separator_vertical.tkraise()

    def change_to_frame_instant_stats(
            self,
    ) -> None:
        """
        Raise frame stats to the top.
        """

        self.frame_instant_stats.tkraise()
        # self.separator_vertical.tkraise()

    def change_to_frame_lifetime_stats(
            self,
    ) -> None:
        """
        Raise frame lifetime stats to the top.
        """

        self.frame_lifetime_stats.tkraise()
        # self.separator_vertical.tkraise()

    def callback_next_allocation(
            self,
    ) -> None:

        if self._idle_job is not None:
            self.after_cancel(self._idle_job)

        # Update user text labels for the new simulation state
        self.update_user_text_labels()

        # Reset user allocated resources memory
        self.resources_per_user = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
        }

        # Clear primary resource grid
        self.frame_scenario.resource_grid.clear()

        # Reset countdown
        self.countdown_value = self.config_gui.countdown_reset_value_seconds

        # Update learned algorithm simulations to copy primary sim state
        self.update_secondary_simulations()

        self.label_explanations.config(text=self.config_gui.label_explanation_text1)

        self.frame_explanation.tkraise()

    def callback_button_next_ai(
            self,
    ) -> None:

        self.frame_allocations.tkraise()

        if self._idle_job is not None:
            self.after_cancel(self._idle_job)
        self._idle_job = self.after(ms=self.config_gui.reset_timeout_ms, func=self.reset)

    def callback_button_toggle_tutorial(
            self,
    ) -> None:
        """
        Toggle tutorial visible/invisible.
        """

        if self.tutorial_toggle:
            self.frame_tutorial.lower()
            self.tutorial_toggle = False
        else:
            self.frame_tutorial.tkraise()
            self.tutorial_toggle = True

    def callback_button_timer(
            self,
    ) -> None:
        """
        Countdown button on-press. Toggle countdown mode. If now active, initialize countdown,
        else indicate that countdown is not active.
        """

        self.countdown_toggle = not self.countdown_toggle
        if self.countdown_toggle:
            self.countdown_value = self.config_gui.countdown_reset_value_seconds
            self.after(1000, self.check_loop)

        if not self.countdown_toggle:
            self.frame_screen_action_buttons.button_countdown.configure(text='Countdown', image=self.tk_image_button_countdown)

    def callback_button_auto_mode(
            self,
    ) -> None:
        """
        Auto mode allocates one resource at random per second while active.
        """

        self.auto_mode_toggle = not self.auto_mode_toggle
        if self.auto_mode_toggle:
            self.frame_screen_action_buttons.button_auto.configure(text='Auto on', image=self.tk_image_button_auto)
            self.after(100, self.auto_mode_allocate)
        else:
            self.frame_screen_action_buttons.button_auto.configure(text='Auto', image=self.tk_image_button_auto)

    def auto_mode_allocate(
            self,
    ) -> None:
        """
        If auto mode toggle is active, allocate one resource, then queue self again after one second.
        """

        if self.auto_mode_toggle:
            random_user_id = self.config.rng.choice(range(sum(self.config.num_users.values())))
            self.allocate_resource(user_id=random_user_id)
            self.after(ms=498, func=self.auto_mode_allocate)

    def callback_button_reset(
            self,
    ) -> None:
        """
        Reset all states and statistics.
        """

        # toggle off auto mode
        if self.auto_mode_toggle:
            self.callback_button_auto_mode()

        # toggle off countdown mode
        if self.countdown_toggle:
            self.callback_button_timer()

        # reset primary resource grid
        self.frame_scenario.resource_grid.clear()

        # Reset last allocation indicator
        empty_allocation = {ue: 0 for ue in range(sum(self.config.num_users.values()))}
        empty_allocation_color_dict = {ue: 'white' for ue in range(sum(self.config.num_users.values()))}
        for allocator in self.config_gui.allocator_names_static:
            self.frame_allocations.resource_grids[allocator].fill(
                allocation=empty_allocation,
                color_dict=empty_allocation_color_dict,
            )

        # reset rigged start states
        self.rigged_start_states = self.config_gui.rigged_start_states.copy()
        if len(self.rigged_start_states) > 0:
            rigged_state = self.rigged_start_states.pop(0)
            self.rig_sim_main_state(state=rigged_state)
            self.update_user_text_labels()
            self.update_secondary_simulations()


    def allocate_resource(
            self,
            user_id,
    ) -> None:
        """
        Allocate one resource to user user_id.
        :param user_id: User to allocate to.
        """

        if self._idle_job is not None:
            self.after_cancel(self._idle_job)
        self._idle_job = self.after(ms=self.config_gui.reset_timeout_ms, func=self.reset)

        # Allocate visually
        current_resource_pointer = self.frame_scenario.resource_grid.allocate(
            color=self.config_gui.user_colors[user_id]
        )

        if current_resource_pointer == 1:
            self.label_explanations.config(
                text=self.config_gui.label_explanation_text2
            )

        # Bookkeeping
        self.resources_per_user[user_id] += 1

        # If all resources allocated -> evaluate
        if current_resource_pointer == self.config.num_total_resource_slots:
            self.after(100, self.evaluate_allocation)  # small delay makes it feel more natural
            self.frame_results_explanation.tkraise()

    def rig_sim_main_state(
            self,
            state: list,
    ) -> None:
        """
        Cheat by setting a specific simulation state.
        :param state: [[user0jobsize, user0powergain], [user1jobsize, user1powergain], ...]
        """

        for user_id, user_state in enumerate(state):
            self.sim_main.users[user_id].generate_specific_job(user_state[0])
            self.sim_main.users[user_id].set_specific_power_gain(user_state[1])

    def evaluate_allocation(
            self,
    ) -> None:
        """
        Evaluate user allocation & learner allocations. Update GUI accordingly.
        """

        self._idle_job = self.after(ms=self.config_gui.reset_timeout_ms, func=self.reset)

        # Convert user allocation to sim expected formatting
        action = np.array(list(self.resources_per_user.values())) / self.config.num_total_resource_slots
        action = action.astype('float32')

        # Fill the small allocation resource grid with user allocation
        self.frame_allocations.resource_grids[self.config_gui.allocator_names_static[0]].fill(
            allocation=self.get_allocated_slots(percentage_allocation_solution=action, sim=self.sim_main),
            color_dict=self.config_gui.user_colors,
        )

        # Calculate achieved metrics
        reward, reward_components = self.sim_main.step(percentage_allocation_solution=action)

        if len(self.rigged_start_states) > 0:
            rigged_state = self.rigged_start_states.pop(0)
            self.rig_sim_main_state(state=rigged_state)

        # Format for table display
        instant_stats = [[
            reward_components['sum rate'],
            reward_components['fairness score'],
            reward_components['prio jobs missed'],
            reward
        ]]

        # Bookkeeping
        # self.lifetime_stats['self']['sumrate'].append(reward_components['sum rate'])
        # self.lifetime_stats['self']['fairness'].append(reward_components['fairness score'])
        # self.lifetime_stats['self']['timeouts'].append(reward_components['prio jobs missed'])
        # self.lifetime_stats['self']['overall'].append(reward)

        # Repeat the same for learned algorithm calculations
        for learner_name, learner in self.config_gui.learned_agents.items():

            # Get allocation action
            action = learner.call(self.secondary_simulations[learner_name].get_state()[np.newaxis]).numpy().squeeze()

            # Fill the small allocation resource grid with user allocation
            self.frame_allocations.resource_grids[self.config_gui.learned_agents_display_names_static[learner_name]].fill(
                allocation=self.get_allocated_slots(percentage_allocation_solution=action, sim=self.secondary_simulations[learner_name]),
                color_dict=self.config_gui.user_colors,
            )

            # Calculate achieved metrics
            reward, reward_components = self.secondary_simulations[learner_name].step(percentage_allocation_solution=action)

            # Format for table display
            instant_stats.append(
                [
                    reward_components['sum rate'],
                    reward_components['fairness score'],
                    reward_components['prio jobs missed'],
                    reward
                ]
            )

            # Bookkeeping
            # self.lifetime_stats[learner_name]['sumrate'].append(reward_components['sum rate'])
            # self.lifetime_stats[learner_name]['fairness'].append(reward_components['fairness score'])
            # self.lifetime_stats[learner_name]['timeouts'].append(reward_components['prio jobs missed'])
            # self.lifetime_stats[learner_name]['overall'].append(reward)

        # Update instant stats
        # self.frame_allocations.instant_stats.clear()
        instant_stats = np.round(np.array(instant_stats), 1)

        results_text = "Ihre Verteilung erreicht:\n\n"
        results_text += f"· Eine Datenrate von {instant_stats[0, 0]} bits/s/Hz. "
        if instant_stats[0, 0] < instant_stats[1, 0]:
            results_text += "Da geht mehr!"
        else:
            results_text += "Gut!"
        results_text += '\n\n'

        results_text += f"· {instant_stats[0, 1]:.0%} Fairness. "
        if instant_stats[0, 1] < instant_stats[1, 1]:
            results_text += "Da geht mehr!"
        else:
            results_text += "Gut!"
        results_text += '\n\n'

        if instant_stats[0, 2] > 0:
            results_text += "· Der Krankenwagen wurde nicht ausreichend versorgt!\n"
        else:
            results_text += "· Der Krankenwagen erhält ausreichend Daten. Gut!\n"

        self.label_results_explanation.config(text=results_text)

        # cmaps = [
        #     self.config_gui.fig_lifetime_stats_gradient_cmap,  # sumrate
        #     self.config_gui.fig_lifetime_stats_gradient_cmap,  # fairness
        #     self.config_gui.fig_lifetime_stats_gradient_cmap_reversed,  # deaths
        #     self.config_gui.fig_lifetime_stats_gradient_cmap,  # overall
        # ]
        # colors = [[[0, 0, 0, 0] for _ in range(4)] for _ in range(2)]
        # for column_index, cmap in enumerate(cmaps):
        #
        #     column_stats = instant_stats[:, column_index].copy()
        #     column_stats += min(column_stats)  # transform to positive space
        #     if max(column_stats) > 0:
        #         column_stats = column_stats / max(column_stats)  # transform to [0, 1]
        #
        #     # set color map
        #     column_colors = cmap(column_stats)
        #
        #     for column_color_id, column_color in enumerate(column_colors):
        #         colors[column_color_id][column_index] = list(column_color)

        # self.frame_allocations.instant_stats.draw_instant_stats_table(data=instant_stats, colors=colors, **self.config_gui.fig_instant_stats_config)

    def get_channel_strength_image(
            self,
            channel_strength,
    ) -> tk.PhotoImage:
        """
        Select a channel strength image based on numerical channel strength
        :param channel_strength: input channel strength
        :return: A tkImage to display
        """

        if channel_strength >= 16:
            return self.tk_images_channel_strength[3]
        if channel_strength >= 9:
            return self.tk_images_channel_strength[2]
        if channel_strength >= 3:
            return self.tk_images_channel_strength[1]
        if channel_strength >= 1:
            return self.tk_images_channel_strength[0]

        raise ValueError('Unexpected channel strength')

    def change_language(
            self,
            language,
    ) -> None:

        if language == 'DE':
            self.config_gui._strings_file = 'strings_de_simple.yml'
            self.config_gui.set_strings()
            self.config_gui.set_config_dicts()

        elif language == 'EN':
            self.config_gui._strings_file = 'strings_en.yml'
            self.config_gui.set_strings()
            self.config_gui.set_config_dicts()

        else:
            raise ValueError(f'unknown language {language}')


        # update scenario
        self.frame_scenario.label_title.configure(text=self.config_gui.label_title_text)
        self.update_user_text_labels()
        self.frame_scenario.label_text_resource_grid.configure(text=self.config_gui.label_resource_grid_text)

        # update buttons
        self.frame_screen_selector.screen_selector_button_lifetime_stats.configure(text=self.config_gui.button_screen_selector_lifetime_stats_text)
        self.frame_screen_selector.screen_selector_button_allocations.configure(text=self.config_gui.button_screen_selector_allocations_text)

        # update screen allocations
        self.frame_allocations.label_results_title.configure(text=self.config_gui.label_results_title_text)
        for allocator_name, label_resource_grid in zip(self.config_gui.allocator_names, self.frame_allocations.labels_resource_grids_title):
            label_resource_grid.configure(text=allocator_name)
        self.frame_allocations.label_instant_stats_title.configure(text=self.config_gui.label_instant_stats_title_text)

        # rename instant stats table
        cells = self.frame_allocations.instant_stats.table_instant_stats._cells
        for celL_index, cell_id in enumerate([(1, -1), (2, -1), (3, -1), (4, -1)]):
            cells[cell_id]._text.set_text(self.config_gui.allocator_names[celL_index])

        for cell_index, cell_id in enumerate([(0, 0), (0, 1), (0, 2), (0, 3)]):
            cells[cell_id]._text.set_text(self.config_gui.strings['stats'][cell_index])

        self.frame_allocations.instant_stats.canvas.draw()

        # rename frame lifetime stats
        self.frame_lifetime_stats.label_title.configure(text=self.config_gui.label_lifetime_stats_title_text)

        for fig_id, fig in enumerate([self.frame_lifetime_stats.fig_throughput, self.frame_lifetime_stats.fig_fairness, self.frame_lifetime_stats.fig_deaths, self.frame_lifetime_stats.fig_overall]):
            fig_title = self.config_gui.strings['stats'][fig_id]
            fig.column_labels = self.config_gui.allocator_names
            fig.ax.set_yticks(range(4), reversed(fig.column_labels), fontsize=fig.font_size)
            fig.ax.set_title(fig_title)
            fig.title = fig_title
            fig.canvas.draw()


    def update_user_text_labels(
            self,
    ) -> None:
        """
        Update user text labels based on the current primary sim state.
        """

        # Resource Wants
        for label_text_user_id, frame_user in enumerate(self.frame_scenario.frames_users):
            if self.sim_main.users[label_text_user_id].job:
                resources = self.sim_main.users[label_text_user_id].job.size_resource_slots
            else:
                resources = 0
            if resources == 1:
                text = f'{self.config_gui.string_wants}: {resources} {self.config_gui.string_resources_singular}'
            else:
                text = f'{self.config_gui.string_wants}: {resources} {self.config_gui.string_resources_plural}'

            frame_user.label_user_text_wants.configure(
                text=text
            )

        # Channel Strength
        for label_text_user_id, frame_user in enumerate(self.frame_scenario.frames_users):
            channel_strength = self.sim_main.users[label_text_user_id].power_gain
            text = f'{self.config_gui.string_channel}: '
            frame_user.label_user_text_channel_strength.configure(
                text=text,
                image=self.get_channel_strength_image(channel_strength),
                compound=tk.RIGHT,
            )

    def update_secondary_simulations(
            self,
    ) -> None:
        """
        Sync all secondary simulations to the primary sim state.
        """

        for sec_sim in self.secondary_simulations.values():
            sec_sim.import_state(state=self.sim_main.export_state())

    def get_allocated_slots(
            self,
            percentage_allocation_solution: np.ndarray,
            sim,
    ) -> dict[int: int]:
        """
        Convert a floating point percentage allocation solution into discrete blocks, same as the sim would do it.
        :param percentage_allocation_solution: [float_user0, float_user1, ...] with sum(.)=1.
        :param sim: The sim to check for total users, user requests, etc.
        :return: dict, Discrete allocations per user.
        """

        requested_slots_per_ue = [
            sim.users[ue_id].job.size_resource_slots if sim.users[ue_id].job else 0
            for ue_id in range(len(sim.users))
        ]

        slot_allocation_solution = [
            np.minimum(
                np.round(percentage_allocation_solution[ue_id] * self.config.num_total_resource_slots),
                requested_slots_per_ue[ue_id],
                dtype='float32'
            )
            for ue_id in range(len(sim.users))
        ]

        # grant at most one additional resource if there was rounding down
        if sum(slot_allocation_solution) == sim.resource_grid.total_resource_slots - 1:
            remainders = np.round([
                percentage_allocation_solution[ue_id] * sim.resource_grid.total_resource_slots - slot_allocation_solution[ue_id]
                for ue_id in range(len(sim.users))
            ], decimals=5)
            for ue_id in range(len(sim.users)):
                if remainders[ue_id] > 0:
                    if requested_slots_per_ue[ue_id] > slot_allocation_solution[ue_id]:
                        slot_allocation_solution[ue_id] += 1
                        break

        # Check if the rounding has resulted in more resources distributed than available
        if sum(slot_allocation_solution) > sim.resource_grid.total_resource_slots:
            # if so, remove one resource from a random user
            while sum(slot_allocation_solution) > sim.resource_grid.total_resource_slots:
                random_user_id = self.config.rng.integers(0, len(sim.users))
                if slot_allocation_solution[random_user_id] > 0:
                    slot_allocation_solution[random_user_id] -= 1

        # Prepare the allocated slots per ue for metrics calculation
        allocated_slots_per_ue: dict = {
            ue_id: slot_allocation_solution[ue_id]
            for ue_id in range(len(sim.users))
        }

        return allocated_slots_per_ue


if __name__ == '__main__':
    app = App()
    app.mainloop()
