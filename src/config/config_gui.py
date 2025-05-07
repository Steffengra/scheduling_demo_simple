
from tkinter import (
    LEFT,
    CENTER,
    TOP,
)
from pathlib import Path

import yaml

# from matplotlib.colors import LinearSegmentedColormap
from keras.models import (
    load_model,
)


class ConfigGUI:

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        self._strings_file = 'strings_de.yml'  # text for all visible strings

        global_font_scale = 1.0  # scales fonts and elements that scale with font size, e.g., boxes
        self.reset_timeout_ms = 5 * 60000

        self.rigged_start_states = [  # [[user0jobsize, user0powergain], [user1jobsize, user1powergain], ...]
            [[3, 9], [4, 1], [1, 1], [3, 16]],
            [[5, 16], [7, 9], [0, 1], [5, 1]],
            [[3, 16], [6, 9], [3, 1], [0, 4]],
        ]

        # images
        self.logos = [
            'unilogo.png',
            'ANT.png',
            'sponsoredbybmbf.png',
            # 'momentum.jpg',
            # 'FunKI_Logo_final_4C.png',
        ]
        self.user_images = [
            'laptop.jpg',
            'taking_picture_crop.jpg',
            'textmessage2.jpg',
            'emergency.jpg',
        ]
        self.channel_strength_indicator_imgs = [
            'bars_low_alt_mono.png',
            'bars_medlow_alt_mono.png',
            'bars_medhigh_alt_mono.png',
            'bars_high_mono.png',
        ]
        self.base_station_image = 'base_station.png'
        self.button_countdown_img = 'stopwatch2.png'
        self.button_auto_img = 'robot2.png'
        self.button_reset_img = 'reset2.png'
        self.flag_images = [
            Path(self.project_root_path, 'src', 'analysis', 'img', 'flag_DE.png'),
            Path(self.project_root_path, 'src', 'analysis', 'img', 'flag_EN.png'),
        ]

        self.label_img_logos_height_scale = 0.07
        self.label_img_user_height_scale = 0.12
        self.label_img_users_border_width = 15
        self.label_img_base_station_height_scale = 0.25
        self.label_img_flag_height_scale = 0.1
        self.label_resource_small_scaling: float = 0.3
        self.button_countdown_img_scale = 0.045

        self.label_resource_width = 3  # relative to font size
        self.label_resource_height = 1  # relative to font size
        self.label_resource_border_width = 2

        # fonts
        self.label_title_font = ('Arial', int(global_font_scale * 60))
        self.label_user_font = ('Arial', int(global_font_scale * 18))
        self.label_resource_font = ('Arial', int(global_font_scale * 50))
        self.button_screen_selector_font = ('Arial', int(global_font_scale * 25))
        self.button_action_font = ('Arial', int(global_font_scale * 25))
        self.table_instant_stats_font_size = int(global_font_scale * 13)
        self.fig_lifetime_stats_font_size = int(global_font_scale * 13)
        self.label_resource_grid_title_font = ('Arial', int(global_font_scale * 15))
        self.label_arena_attribution_font = ('Arial', int(global_font_scale * 10))

        # colors
        self.fig_lifetime_stats_bar_colors_positive = [self.cp3['blue3'], self.cp3['blue3']]
        self.fig_lifetime_stats_bar_colors_negative = [self.cp3['red3'], self.cp3['red3']]
        self.table_instant_stats_color_gradient = [self.cp3['red2'], self.cp3['blue3'], self.cp3['blue2']]
        self.user_colors = {
            0: self.cp3['blue1'],
            1: self.cp3['blue2'],
            2: self.cp3['blue3'],
            3: self.cp3['red2'],
        }

        self.countdown_reset_value_seconds: int = 10

        self.learned_agents: dict = {
            # 'sumrate': load_model(Path(self.models_path, 'max_sumrate', 'policy')),
            # 'fairness': load_model(Path(self.models_path, 'fairness', 'policy_snap_0.914')),
            'mixed': load_model(Path(self.models_path, 'mixed', 'policy_snap_1.020')),
        }

        self._post_init()

    def _pre_init(
            self,
    ) -> None:

        self.project_root_path = Path(__file__).parent.parent.parent
        self.models_path = Path(self.project_root_path, 'models')

        self._load_palettes()

    def _post_init(
            self,
    ) -> None:

        self.set_strings()

        # self.allocator_names_static = [self.own_allocation_display_name] + list(self.learned_agents_display_names.values())  # static once loaded
        self.allocator_names_static = ['Sie', 'KI ']
        self.learned_agents_display_names_static = self.strings['label_learned_display_names']

        self.set_config_dicts()

    def set_config_dicts(
            self,
    ) -> None:

        # self.allocator_names = [self.own_allocation_display_name] + list(self.learned_agents_display_names.values())
        self.allocator_names = ['Sie', 'KI ']

        # buttons

        self.button_tutorial_config = {
            'text': '?',
            'font': self.button_screen_selector_font,
            'width': 1,
            'height': 1,
            'borderwidth': 0,
            'bg': 'white',
            'compound': CENTER,
        }

        self.button_next_config = {
            'text': 'Nochmal',
            'font': self.button_screen_selector_font,
            # 'width': 1,
            # 'height': 1,
            'borderwidth': 0,
            'bg': 'white',
            'compound': CENTER,
        }

        self.button_next_ai_config = {
            'text': 'Weiter',
            'font': self.button_screen_selector_font,
            # 'width': 1,
            # 'height': 1,
            'borderwidth': 0,
            'bg': 'white',
            'compound': CENTER,
        }

        self.button_screen_selector_config = {
            'font': self.button_screen_selector_font,
            'width': 1,
            'borderwidth': 3,
            'bg': 'white',
            'compound': CENTER,
        }

        self.button_action_config = {
            'font': self.button_action_font,
            'width': 1,  # small value for even distribution
            'borderwidth': 7,
            'bg': 'white',
            'compound': TOP,
        }

        self.button_screen_selector_allocations_config = {
            'text': self.button_screen_selector_allocations_text,
            **self.button_screen_selector_config,
        }

        self.button_screen_selector_lifetime_stats_config = {
            'text': self.button_screen_selector_lifetime_stats_text,
            **self.button_screen_selector_config,
        }

        self.button_countdown_config = {
            'text': 'Countdown',
            **self.button_action_config,
        }

        self.button_auto_config = {
            'text': 'Auto',
            **self.button_action_config,
        }

        self.button_reset_config = {
            'text': 'Reset',
            **self.button_action_config,
        }

        # labels
        self.labels_config = {
            'font': self.label_user_font,
            'bg': 'white',
        }

        self.label_resource_config = {
            'font': self.label_resource_font,
            'width': self.label_resource_width,
            'height': self.label_resource_height,
            'relief': 'solid',
            'borderwidth': self.label_resource_border_width,
            'bg': 'white',
        }

        self.label_resource_small_config = self.label_resource_config.copy()
        self.label_resource_small_config['font'] = (self.label_resource_font[0], int(self.label_resource_small_scaling * self.label_resource_font[1]))

        self.label_user_image_config = {
            'bg': 'white',
            'highlightthickness': self.label_img_users_border_width,
        }

        self.label_arena_attribution_config = {
            'font': self.label_arena_attribution_font,
            # 'wraplength': 100,
            'bg': 'white',
        }

        self.label_resource_grid_title_config = {
            'font': self.label_resource_grid_title_font,
            'height': 3,
            'wraplength': 100,
            'bg': 'white',
        }

        self.label_user_text_config = {
            'font': self.label_user_font,
            'bg': 'white',
            'justify': LEFT,
        }

        self.label_resource_grid_text_config = {
            'text': self.label_resource_grid_text,
            'font': self.button_screen_selector_font,
            'bg': 'white',
            'justify': LEFT,
        }

        self.label_title_text_config = {
            'text': self.label_title_text,
            'font': self.label_title_font,
            'bg': 'white',
            'justify': LEFT,
        }

        self.label_explanations_text_config = {
            'font': self.button_screen_selector_font,
            'bg': 'white',
            'justify': LEFT,
        }

        self.label_results_title_config = {
            'text': self.label_results_title_text,
            **self.labels_config,
        }

        self.label_instant_stats_title_config = {
            'text': self.label_instant_stats_title_text,
            'font': self.label_user_font,
            'bg': 'white',
        }

        self.label_lifetime_stats_title_config = {
            'text': self.label_lifetime_stats_title_text,
            'font': self.label_user_font,
            'bg': 'white',
        }

        # frames
        self.frames_config = {
            'bg': 'white',
            # 'relief': 'solid',
            # 'borderwidth': 2,
        }

        self.fig_instant_stats_config = {
            'column_labels': self.stat_names,
            'row_labels': self.allocator_names,
            'font_size': self.table_instant_stats_font_size,
        }

        self.fig_lifetime_stats_bars_config = {
            'column_labels': self.allocator_names,
            'font_size': self.fig_lifetime_stats_font_size,
        }

        self.fig_lifetime_stats_bars_throughput_config = {
            'title': self.strings['stats'][0],
            'bar_colors': self.fig_lifetime_stats_bar_colors_positive,
            'xlim_max_initial': 100,
            **self.fig_lifetime_stats_bars_config,
        }

        self.fig_lifetime_stats_bars_fairness_config = {
            'title': self.strings['stats'][1],
            'bar_colors': self.fig_lifetime_stats_bar_colors_positive,
            'xlim_max_initial': 4,
            **self.fig_lifetime_stats_bars_config,
        }

        self.fig_lifetime_stats_bars_deaths_config = {
            'title': self.strings['stats'][2],
            'bar_colors': self.fig_lifetime_stats_bar_colors_negative,
            'xlim_max_initial': 4,
            **self.fig_lifetime_stats_bars_config,
        }

        self.fig_lifetime_stats_bars_overall_config = {
            'title': self.strings['stats'][3],
            'bar_colors': self.fig_lifetime_stats_bar_colors_positive,
            'xlim_max_initial': 4,
            **self.fig_lifetime_stats_bars_config,
        }

        # self.fig_lifetime_stats_gradient_cmap = LinearSegmentedColormap.from_list('', self.table_instant_stats_color_gradient)
        # self.fig_lifetime_stats_gradient_cmap_reversed = LinearSegmentedColormap.from_list('', list(reversed(self.table_instant_stats_color_gradient)))

    def set_strings(
            self,
    ) -> None:

        with open(Path(self.project_root_path, 'src', 'config', self._strings_file), 'r') as file:
            self.strings = yaml.safe_load(file)

        self.button_screen_selector_allocations_text = self.strings['button_screen_selector_allocations']
        self.button_screen_selector_lifetime_stats_text = self.strings['button_screen_selector_lifetime_stats']

        self.label_title_text = self.strings['label_title']
        self.label_resource_grid_text = self.strings['label_resource_grid']
        self.label_instant_stats_title_text = self.strings['label_instant_stats_title']
        self.label_lifetime_stats_title_text = self.strings['label_lifetime_stats_title']
        self.stat_names = self.strings['stats']
        self.own_allocation_display_name = self.strings['label_own_display_name']
        self.learned_agents_display_names = self.strings['label_learned_display_names']
        self.label_results_title_text = self.strings['label_results_title']

        self.string_wants = self.strings['wants']
        self.string_resources_singular = self.strings['resources_singular']
        self.string_resources_plural = self.strings['resources_plural']
        self.string_channel = self.strings['channel']

        self.string_arena_attribution = self.strings['arena_attribution']

        self.label_explanation_text1 = """Stellen Sie sich vor: Sie sind im Stadion und möchten einen Schnapschuss Ihrer Lieblingsband verschicken - zusammen mit 50.000 anderen Besuchern!
Nicht nur in solchen Ausnahmesituationen muss das Kommunikationsnetz entscheiden, welche Daten zuerst über das geteilte Funkmedium gesendet werden.
        
Tippen Sie auf einen Nutzer, um einen Frequenzblock zuzuteilen."""
        self.label_explanation_text2 = """Häufig sind Funkressourcen begrenzt und nicht alle Nutzer können gleichzeitig senden.

Typische Verteilungskriterien sind dabei:
· Datenrate: Wie viele Daten können gleichzeitig fehlerfrei übertragen werden? Ein besserer Empfang erlaubt mehr Daten.
· Fairness: Können alle etwa gleich viele Daten senden, auch bei schlechtem Empfang?
· Risiko: Sind einige Nutzer zu priorisieren?

Diese Ziele werden gegeneinander balanciert.
Verteilen Sie nun die restlichen Frequenzblöcke an die Nutzer."""

        self.label_explanation_textai = "Die Zuteilung muss nicht nur sinnvoll, sondern auch sehr schnell erfolgen. Künstliche Intelligenz (KI) kann dabei helfen. Forschung hilft, dass man sich auf die Entscheidungen der KI verlassen kann.\n\nEine KI, die versucht alle Ziele zu balancieren, hätte etwa so verteilt:"
#Häufig sind Funkressourcen begrenzt und nicht alle Nutzer können gleichzeitig senden. KI kann helfen, die Ressourcen sinnvoll zu verteilen.

#Typische Verteilungskriterien sind dabei:
#- Datenrate: Wie viele bits kann ich übertragen?
#- Fairness: Können alle etwa gleich senden, auch bei schlechterem Kanal?
#- Risiko: Sind einige Nutzer zu priorisieren?

#Diese Ziele werden gegeneinander balanciert. Forschung hilft,
# KI kann helfen, die Ressourcen sinnvoll zu verteilen.
# Forschung hilft, die KI so zu steuern, dass man sich auf ihre Entscheidungen verlassen kann.
#"""

    def _load_palettes(
            self,
    ) -> None:

        self.cp3: dict[str: str] = {  # uni branding
            'red1': '#9d2246',
            'red2': '#d50c2f',
            'red3': '#f39ca9',
            'blue1': '#00326d',
            'blue2': '#0068b4',
            'blue3': '#89b4e1',
            'purple1': '#3b296a',
            'purple2': '#8681b1',
            'purple3': '#c7c1e1',
            'peach1': '#d45b65',
            'peach2': '#f4a198',
            'peach3': '#fbdad2',
            'orange1': '#f7a600',
            'orange2': '#fece43',
            'orange3': '#ffe7b6',
            'green1': '#008878',
            'green2': '#8acbb7',
            'green3': '#d6ebe1',
            'yellow1': '#dedc00',
            'yellow2': '#f6e945',
            'yellow3': '#fff8bd',
            'white': '#ffffff',
            'black': '#000000',
        }
