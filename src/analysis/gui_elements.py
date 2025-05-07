
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import src
from src.utils.get_width_rescale_constant_aspect_ratio import (
    get_width_rescale_constant_aspect_ratio,
)


class Scenario(tk.Frame):
    """
    Left hand side of the screen, holds logos, users, their wants, the primary resource grid.
    """

    def __init__(
            self,
            window_width: int,
            window_height: int,
            config_gui: 'src.config.config_gui.ConfigGUI',
            logo_paths: list[Path],
            user_image_paths: list[Path],
            base_station_image_path: Path,
            button_callbacks: list,
            language_button_callbacks: list,
            tutorial_button_callback,
            num_total_resource_slots: int,
            **kwargs,
    ) -> None:

        super().__init__(width=1.0 * window_width, height=1.0 * window_height, **kwargs)

        self.logo_img_height = int(config_gui.label_img_logos_height_scale * window_height)
        self.user_img_height = int(config_gui.label_img_user_height_scale * window_height)
        self.base_station_img_height = int(config_gui.label_img_base_station_height_scale * window_height)
        self.flag_img_height = int(config_gui.label_img_flag_height_scale * window_height)

        # Logo Bar
        self.frame_logos = tk.Frame(master=self, **config_gui.frames_config)
        self._logo_setup(logo_paths=logo_paths, label_user_text_config=config_gui.label_user_text_config)

        # Title Line
        self.label_title = tk.Label(self, **config_gui.label_title_text_config)


        # Scenario Objects
        self.frame_arena = BaseStationFrame(
            master=self,
            base_station_image_path=Path(base_station_image_path.parent, '20250507_A_large_stadium.png'),
            base_station_img_height=int(3*self.base_station_img_height),
            **config_gui.frames_config,
        )
        self.label_arena_attribution = tk.Label(
            master=self,
            text=config_gui.string_arena_attribution,
            **config_gui.label_arena_attribution_config,
        )
        # self.frame_base_station = BaseStationFrame(
        #     master=self,
        #     base_station_image_path=base_station_image_path,
        #     base_station_img_height=self.base_station_img_height,
        #     **config_gui.frames_config,
        # )
        self.frames_users = [
            UserFrame(
                master=self,
                button_callback=button_callbacks[user_id],
                user_image_path=user_image_paths[user_id],
                user_img_height=self.user_img_height,
                user_color=config_gui.user_colors[user_id],
                label_user_img_config=config_gui.label_user_image_config,
                label_user_text_config=config_gui.label_user_text_config,
                **config_gui.frames_config
            )
            for user_id in range(len(config_gui.user_images))
        ]

        # Primary Resource Grid
        self.frame_resource_grid = tk.Frame(master=self, width=.15 * window_width, height=0.9 * window_height, **config_gui.frames_config)
        self.subframe_resource_grid = tk.Frame(master=self.frame_resource_grid, **config_gui.frames_config)

        self.label_text_resource_grid = tk.Label(self.subframe_resource_grid, **config_gui.label_resource_grid_text_config)
        self.resource_grid = ResourceGrid(self.subframe_resource_grid, config_gui.label_resource_config, num_total_resource_slots)

        # self.button_de = LanguageFlagButton(
        #     flag_image=config_gui.flag_images[0],
        #     flag_image_img_height=self.flag_img_height,
        #     button_callback=language_button_callbacks[0],
        # )
        # self.button_en = LanguageFlagButton(
        #     flag_image=config_gui.flag_images[1],
        #     flag_image_img_height=self.flag_img_height,
        #     button_callback=language_button_callbacks[1],
        # )

        # self.button_tutorial_frame = tk.Frame()
        # self.label_tutorial_button = tk.Button(
        #     self.button_tutorial_frame,
        #     command=tutorial_button_callback,
        #     **config_gui.button_tutorial_config,
        # )

        self._place_items()

    def _logo_setup(
            self,
            logo_paths: list[Path],
            label_user_text_config: dict,
    ) -> None:

        self.images_logos = [
            Image.open(logo_path)
            for logo_path in logo_paths
        ]

        self.tk_image_logos = [
            ImageTk.PhotoImage(image_logo.resize((
                get_width_rescale_constant_aspect_ratio(image_logo, self.logo_img_height),
                self.logo_img_height,
            )))
            for image_logo in self.images_logos
        ]

        self.labels_img_logos = [
            tk.Label(self.frame_logos, image=tk_image_logo, **label_user_text_config)
            for tk_image_logo in self.tk_image_logos
        ]

    def _place_items(
            self,
    ) -> None:

        self.frame_logos.place(relx=0.0, rely=0.0)
        for label_img_logo in self.labels_img_logos:
            label_img_logo.pack(side=tk.LEFT, padx=10, pady=10)

        self.frame_arena.place(relx=0.4, rely=0.23)
        self.label_arena_attribution.place(relx=.41, rely=.97)

        self.label_title.place(relx=0.02, rely=0.12)

        self.frames_users[0].place(relx=0.42, rely=0.30)
        self.frames_users[1].place(relx=0.66, rely=0.30)
        self.frames_users[2].place(relx=0.42, rely=0.69)
        self.frames_users[3].place(relx=0.66, rely=0.69)

        self.frame_resource_grid.place(relx=0.85, rely=0.1)
        self.frame_resource_grid.pack_propagate(False)

        self.subframe_resource_grid.pack(expand=True)
        self.label_text_resource_grid.pack(side=tk.TOP, pady=10)
        self.resource_grid.place()

        # self.button_de.place(relx=0.00, rely=0.9)
        # self.button_en.place(relx=0.06, rely=0.9)
        # self.button_tutorial_frame.place(relx=1-0.03, rely=0.01)


class UserFrame(tk.Frame):
    """
    Frame for a user with an image and text.
    """

    def __init__(
            self,
            button_callback,
            user_image_path: Path,
            user_img_height: int,
            user_color: str,
            label_user_img_config: dict,
            label_user_text_config: dict,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.user_image = Image.open(user_image_path)
        self.user_image = self.user_image.resize((
            get_width_rescale_constant_aspect_ratio(self.user_image, user_img_height),
            user_img_height,
        ))
        self.user_tk_image = ImageTk.PhotoImage(self.user_image)

        self.label_user_img = tk.Label(
            self,
            image=self.user_tk_image,
            highlightbackground=user_color,
            **label_user_img_config,
        )
        self.label_user_img.bind('<Button-1>', button_callback)

        self.label_user_text_wants = tk.Label(self, **label_user_text_config)
        self.label_user_text_channel_strength = tk.Label(self, **label_user_text_config)

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.label_user_img.pack(pady=10)
        self.label_user_text_wants.pack()
        self.label_user_text_channel_strength.pack()


class LanguageFlagButton(tk.Frame):

    def __init__(
            self,
            flag_image: Path,
            flag_image_img_height: int,
            button_callback,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.flag_image = Image.open(flag_image)
        self.flag_image = self.flag_image.resize((
            get_width_rescale_constant_aspect_ratio(self.flag_image, flag_image_img_height),
            flag_image_img_height,
        ))
        self.flag_tk_image = ImageTk.PhotoImage(self.flag_image)
        self.label_flag_img = tk.Label(
            self,
            image=self.flag_tk_image,
            bg='white',
        )
        self.label_flag_img.bind('<Button-1>', button_callback)

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.label_flag_img.pack()


class BaseStationFrame(tk.Frame):
    """
    Frame for a base station, just an image.
    """

    def __init__(
            self,
            base_station_image_path: Path,
            base_station_img_height: int,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.base_station_image = Image.open(base_station_image_path)
        self.base_station_image = self.base_station_image.resize((
            get_width_rescale_constant_aspect_ratio(self.base_station_image, base_station_img_height),
            base_station_img_height,
        ))
        self.base_station_tk_image = ImageTk.PhotoImage(self.base_station_image)
        self.label_base_station_img = tk.Label(
            self,
            image=self.base_station_tk_image,
            bg='white',
        )

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.label_base_station_img.pack()


class ScreenSelector(tk.Frame):
    """
    Buttons to switch what is displayed on the right hand side of the screen.
    """

    def __init__(
            self,
            config_gui: 'src.config.config_gui.ConfigGUI',
            window_width: int,
            window_height: int,
            button_commands: list,
            **kwargs,
    ) -> None:

        super().__init__(width=0.3 * window_width, height=0.1 * window_height, **kwargs)

        self.screen_selector_button_allocations = tk.Button(
            self,
            command=button_commands[0],
            **config_gui.button_screen_selector_allocations_config,
        )
        self.screen_selector_button_lifetime_stats = tk.Button(
            self,
            command=button_commands[2],
            **config_gui.button_screen_selector_lifetime_stats_config,
        )

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.screen_selector_button_allocations.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.screen_selector_button_lifetime_stats.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


class ScreenActionButtons(tk.Frame):
    """
    Buttons that cause an action, e.g., countdown, reset, ...
    """

    def __init__(
            self,
            config_gui: 'src.config.config_gui.ConfigGUI',
            window_width: int,
            window_height: int,
            button_commands: list,
            **kwargs,
    ) -> None:

        super().__init__(width=0.3 * window_width, height=0.1 * window_height, **kwargs)

        self.button_countdown = tk.Button(
            self,
            command=button_commands[0],
            **config_gui.button_countdown_config,
        )
        self.button_auto = tk.Button(
            self,
            command=button_commands[1],
            **config_gui.button_auto_config,
        )
        self.button_reset = tk.Button(
            self,
            command=button_commands[2],
            **config_gui.button_reset_config,
        )

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.button_countdown.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.button_auto.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.button_reset.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


class ScreenAllocations(tk.Frame):
    """
    One choice of frame for the right hand side of screen. Compares the last allocations of
    all agents (user + learned agents), and their immediate results.
    """

    def __init__(
            self,
            window_height: int,
            window_width: int,
            config_gui: 'src.config.config_gui.ConfigGUI',
            pixels_per_inch: int,
            num_total_resource_slots: int,
            callback_next_allocation,
            **kwargs,
    ) -> None:

        super().__init__(width=.35 * window_width, height=0.9 * window_height, **kwargs)


        # Frame to hold frame title & resource grids frame
        self.frame_allocations = tk.Frame(master=self, height=0.7 * window_height, **config_gui.frames_config)

        self.label_explanation = tk.Label(master=self.frame_allocations, wraplength=.32*window_width, text=config_gui.label_explanation_textai, **config_gui.label_explanations_text_config)

        # Frame to hold all schedulers' resource grids
        self.subframe_allocations = tk.Frame(master=self.frame_allocations, **config_gui.frames_config)

        # Title label in allocations frame
        # self.label_results_title = tk.Label(self.frame_allocations, **config_gui.label_results_title_config)

        # Frames to hold one resource grid for each scheduler
        self.subframes_allocations = {
            allocator_name: tk.Frame(master=self.subframe_allocations, **config_gui.frames_config)
            for allocator_name in config_gui.allocator_names_static
        }

        # Resource Grids for each scheduler
        self.resource_grids = {
            allocator_name: ResourceGridHorizontal(
                master=allocator_subframe,
                label_config=config_gui.label_resource_small_config,
                num_total_resource_slots=num_total_resource_slots,
                )
            for allocator_name, allocator_subframe in self.subframes_allocations.items()
        }
        # Titles for each grid
        self.labels_resource_grids_title = [
            tk.Label(allocator_subframe, text=allocator_name, **config_gui.label_resource_grid_title_config)
            for allocator_name, allocator_subframe in self.subframes_allocations.items()
        ]

        # Frame to hold instant statistics for the last allocation
        # self.frame_instant_stats = tk.Frame(master=self, **config_gui.frames_config)

        # self.label_instant_stats_title = tk.Label(self.frame_instant_stats, **config_gui.label_instant_stats_title_config)
        # self.instant_stats = FigInstantStatsTable(self.frame_instant_stats, fig_width=0.3 * window_width / pixels_per_inch, table_config=config_gui.fig_instant_stats_config)

        self.button_next_frame = tk.Frame(self)
        self.label_button_next_frame = tk.Button(
            self.button_next_frame,
            command=callback_next_allocation,
            **config_gui.button_next_config,
        )

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.frame_allocations.pack()
        self.label_explanation.pack()
        # self.label_results_title.pack(expand=True, ipady=5)
        self.subframe_allocations.pack(expand=True)
        for subframe_allocation in self.subframes_allocations.values():
            subframe_allocation.pack(side=tk.TOP)
        # self.frame_instant_stats.pack(expand=True)

        for label_resource_grid_title in self.labels_resource_grids_title:
            label_resource_grid_title.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        for resource_grid in self.resource_grids.values():
            resource_grid.place()

        # self.label_instant_stats_title.pack()
        # self.instant_stats.place()

        self.label_button_next_frame.pack()
        # self.button_next_frame.place(relx=.8, rely=.9)
        self.button_next_frame.pack()


class ScreenLifetimeStats(tk.Frame):
    """
    One choice of frame for the right hand side of screen. Holds detailed information about lifetime sum stats.
    """

    def __init__(
            self,
            window_width: int,
            window_height: int,
            config_gui: 'src.config.config_gui.ConfigGUI',
            pixels_per_inch: int,
            **kwargs,
    ) -> None:

        super().__init__(width=.3 * window_width, height=0.8 * window_height, **kwargs)

        # Frames
        self.frame_throughput = tk.Frame(master=self, **config_gui.frames_config)
        self.frame_fairness = tk.Frame(master=self, **config_gui.frames_config)
        self.frame_deaths = tk.Frame(master=self, **config_gui.frames_config)
        self.frame_overall = tk.Frame(master=self, **config_gui.frames_config)

        self.label_title = tk.Label(self, **config_gui.label_lifetime_stats_title_config)

        # Fig Lifetime Stats
        # self.fig_throughput = FigLifetimeStatsBars(master=self.frame_throughput, fig_width=0.3*window_width/pixels_per_inch, **config_gui.fig_lifetime_stats_bars_throughput_config)
        # self.fig_fairness = FigLifetimeStatsBars(master=self.frame_fairness, fig_width=0.3*window_width/pixels_per_inch, **config_gui.fig_lifetime_stats_bars_fairness_config)
        # self.fig_deaths = FigLifetimeStatsBars(master=self.frame_deaths, fig_width=0.3*window_width/pixels_per_inch, **config_gui.fig_lifetime_stats_bars_deaths_config)
        # self.fig_overall = FigLifetimeStatsBars(master=self.frame_overall, fig_width=0.3*window_width/pixels_per_inch, **config_gui.fig_lifetime_stats_bars_overall_config)



        # self.label_lifetime_stats_title = tk.Label(self.frame_lifetime_stats, **config_gui.label_lifetime_stats_title_config)
        # self.lifetime_stats = FigLifetimeStats(master=self.frame_lifetime_stats, fig_width=0.3*window_width/pixels_per_inch, **config_gui.fig_lifetime_stats_config)
        #
        # # Fig Instant Stats
        # self.label_instant_stats_title = tk.Label(self.frame_instant_stats, **config_gui.label_instant_stats_title_config)
        # self.instant_stats = FigInstantStatsTable(self.frame_instant_stats, fig_width=0.3 * window_width / pixels_per_inch, table_config=config_gui.fig_instant_stats_config)

        self._place_items()

    def _place_items(
            self,
    ) -> None:

        self.label_title.pack(expand=True)
        self.frame_throughput.pack(expand=True)
        self.frame_fairness.pack(expand=True)
        self.frame_deaths.pack(expand=True)
        self.frame_overall.pack(expand=True)
        # self.fig_throughput.place()
        # self.fig_fairness.place()
        # self.fig_deaths.place()
        # self.fig_overall.place()


class ResourceGrid:
    """
    A resource grid made from vertically stacked boxes that can be colored.
    """

    def __init__(
            self,
            master: tk.Frame,
            label_config: dict,
            num_total_resource_slots: int,
    ) -> None:

        self.pointer: int = 0

        self.labels = [
            tk.Label(master, text='', **label_config)
            for _ in range(num_total_resource_slots)
        ]

    def place(
            self,
    ) -> None:

        for label in self.labels:
            label.pack(side=tk.TOP)

    def allocate(
            self,
            color,
    ) -> int:
        """
        Color the resource succeeding the last accessed resource.
        :param color: Which color
        :return: Index of the colored resource.
        """

        if self.pointer >= len(self.labels):
            return self.pointer+1

        self.labels[self.pointer].config(bg=color)
        self.pointer += 1

        return self.pointer

    def fill(
            self,
            allocation: dict,
            color_dict: dict,
    ) -> None:
        """
        Fill the entire grid with colors according to a given allocation.
        :param allocation: dict[user_id: num_resources]
        :param color_dict:  dict[user_id: color]
        """

        self.clear()

        for user_id, number_of_resources in allocation.items():
            for _ in range(int(number_of_resources)):
                self.allocate(color=color_dict[user_id])

    def clear(
            self,
    ) -> None:
        """
        Color the entire grid white. Reset starting index.
        """

        for label in self.labels:
            label.config(bg='white')

        self.pointer = 0

class ResourceGridHorizontal:
    """
    A resource grid made from vertically stacked boxes that can be colored.
    """

    def __init__(
            self,
            master: tk.Frame,
            label_config: dict,
            num_total_resource_slots: int,
    ) -> None:

        self.pointer: int = 0

        self.labels = [
            tk.Label(master, text='', **label_config)
            for _ in range(num_total_resource_slots)
        ]

    def place(
            self,
    ) -> None:

        for label in self.labels:
            label.pack(side=tk.LEFT)

    def allocate(
            self,
            color,
    ) -> int:
        """
        Color the resource succeeding the last accessed resource.
        :param color: Which color
        :return: Index of the colored resource.
        """

        if self.pointer >= len(self.labels):
            return self.pointer+1

        self.labels[self.pointer].config(bg=color)
        self.pointer += 1

        return self.pointer

    def fill(
            self,
            allocation: dict,
            color_dict: dict,
    ) -> None:
        """
        Fill the entire grid with colors according to a given allocation.
        :param allocation: dict[user_id: num_resources]
        :param color_dict:  dict[user_id: color]
        """

        self.clear()

        for user_id, number_of_resources in allocation.items():
            for _ in range(int(number_of_resources)):
                self.allocate(color=color_dict[user_id])

    def clear(
            self,
    ) -> None:
        """
        Color the entire grid white. Reset starting index.
        """

        for label in self.labels:
            label.config(bg='white')

        self.pointer = 0


class FigInstantStatsTable:
    """
    Figure that holds a table to display metrics of the most recent allocation.
    """

    def __init__(
            self,
            master: tk.Frame,
            fig_width: float,
            table_config: dict,
    ) -> None:

        self.fig = plt.Figure(figsize=(fig_width, 0.32*fig_width))
        self.ax = self.fig.add_subplot()
        self.ax.axis('tight')
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)

        data = np.array([[0.0] * 4] * 2)
        self.fig.tight_layout()
        self.draw_instant_stats_table(data=data, **table_config)
        self.fig.tight_layout()

    def place(
            self,
    ) -> None:

        self.canvas.get_tk_widget().pack(expand=True)

    def clear(
            self,
    ) -> None:

        self.ax.tables[0].remove()

    def draw_instant_stats_table(
            self,
            data: np.ndarray,
            column_labels: list[str],
            row_labels: list[str],
            font_size: int,
            colors: list = None,
    ) -> None:

        if not colors:
            colors = [['white']*4]*2

        self.table_instant_stats = self.ax.table(
            cellText=data,
            cellColours=colors,
            colLabels=column_labels,
            rowLabels=row_labels,
            rowLoc='right',
            loc='center',
            # edges='LR',
        )
        self.table_instant_stats.auto_set_font_size(False)
        self.table_instant_stats.set_fontsize(font_size)
        self.table_instant_stats.scale(xscale=1.2, yscale=1.9)  # scale cell boundaries

        self.canvas.draw()


class FigLifetimeStatsBars:
    """Displays lifetime stats for one particular metric in a stacked bar chart"""

    def __init__(
            self,
            master: tk. Frame,
            fig_width: float,
            column_labels: list[str],
            font_size: int,
            bar_colors: str,
            xlim_max_initial: float,
            title: str,
    ) -> None:

        self.bar_colors = bar_colors
        self.bar_color_toggle = False
        self.xlim_max_initial = xlim_max_initial
        self.xlim_max_current = xlim_max_initial
        self.column_labels = column_labels
        self.font_size = font_size
        self.title = title

        self.fig, self.ax = plt.subplots(figsize=(fig_width, 0.3*fig_width))
        self._fig_setup()

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()

    def _fig_setup(
            self,
    ) -> None:
        t = range(4)
        self.left = np.zeros(4)
        self.last_bars = self.ax.barh(
            y=t,
            width=np.zeros(4),
            height=0.8,
            left=self.left,
            edgecolor='black',
        )

        self.ax.set_xlim([0, self.xlim_max_initial])
        self.ax.set_yticks(range(len(t)), reversed(self.column_labels), fontsize=self.font_size)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_title(self.title, fontsize=self.font_size)
        self.fig.tight_layout()

    def place(
            self,
    ) -> None:
        self.canvas.get_tk_widget().pack(side=tk.TOP)

    def update(
            self,
            new_values,
    ) -> None:

        new_values_reversed = np.array(list(reversed(new_values)))

        t = range(2)
        current_bars = self.ax.barh(
            y=t,
            width=np.where(new_values_reversed > 0, new_values_reversed, 0),
            height=0.8,
            left=self.left,
            color=self.bar_colors[0] if self.bar_color_toggle else self.bar_colors[1],
            edgecolor=['white' if new_values_reversed[entry_id] < 0 else 'black' for entry_id in range(2)],
        )

        # shorten previous bar if negative value
        # todo: there will be some inaccuracies here if current negative is greater than last bar positive
        for new_value_id, new_value in enumerate(new_values_reversed):
            if new_value < 0:
                last_width = self.last_bars[new_value_id].get_width()

                self.last_bars[new_value_id].set_width(max(0, last_width + new_value))

        self.last_bars = current_bars

        self.left += new_values_reversed
        self.bar_color_toggle = not self.bar_color_toggle

        # rescale axes
        if any(self.left > self.xlim_max_current):
            self.xlim_max_current = max(self.left) + 0.5 * max(new_values_reversed)
            self.ax.set_xlim([0, self.xlim_max_current])

        self.canvas.draw()

    def clear(
            self,
    ) -> None:

        self.ax.clear()
        self._fig_setup()
        self.canvas.draw()


class FigLifetimeStats:
    """
    Bar chart of the average overall performance of each allocator.
    """

    def __init__(
            self,
            master: tk.Frame,
            fig_width: float,
            column_labels: list[str],
            font_size: int,
            bar_color: str,
    ) -> None:

        self.maximum_value: float = 0.1  # initialize arbitrarily

        fig = plt.Figure(figsize=(fig_width, 0.4*fig_width))
        self.ax = fig.add_subplot()
        t = range(4)
        self.bars_primary = self.ax.barh(
            t,
            width=[0, 0, 0, 0],
            height=0.8,
            color=bar_color,
            edgecolor='black',
        )
        self.ax.set_xlim([0, 40])
        self.ax.set_yticks(range(len(t)), reversed(column_labels), fontsize=font_size)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.draw()

    def place(
            self,
    ) -> None:

        self.canvas.get_tk_widget().pack(side=tk.TOP)

    def update(
            self,
            values: list,
    ) -> None:

        # Rescale y axis
        for value in values:
            if value > self.maximum_value:
                self.maximum_value = value
        self.ax.set_xlim([0, self.maximum_value * 1.05])

        # Update bars
        for bar, value in zip(reversed(self.bars_primary), values):
            bar.set_width(value)

        self.canvas.draw()
