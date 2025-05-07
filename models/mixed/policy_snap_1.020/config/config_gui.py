
from tkinter import (
    LEFT,
)
from pathlib import (
    Path,
)
from keras.models import (
    load_model,
)


class ConfigGUI:

    def __init__(
            self,
    ) -> None:
        self._pre_init()

        self.label_img_users_height_scale = 0.1
        self.label_img_logos_height_scale = 0.07
        self.label_img_users_border_width = 15

        self.button_font = ('Arial', 50)
        self.button_user_width = 6  # relative to font size
        self.button_user_height = 2  # relative to font size

        self.label_resource_font = ('Arial', 50)
        self.label_resource_width = 3  # relative to font size
        self.label_resource_height = 1  # relative to font size
        self.label_resource_border_width = 2

        self.label_user_font = ('Arial', 25)

        self.countdown_reset_value_seconds: int = 10

        self.user_colors = {
            0: self.cp3['blue1'],
            1: self.cp3['blue2'],
            2: self.cp3['blue3'],
            3: self.cp3['red2'],
        }

        self.learned_agents: dict = {
            'sumrate': load_model(Path(self.models_path, 'max_sumrate', 'policy')),
            'fairness': load_model(Path(self.models_path, 'fairness', 'policy_snap_0.914')),
            'mixed': load_model(Path(self.models_path, 'mixed', 'policy_snap_0.959')),
            # 'sumrate': 1,
            # 'fairness': 1,
            # 'mixed': 1,
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

        self.button_user_config = {
            'font': self.button_font,
            'width': self.button_user_width,
            'height': self.button_user_height,
        }

        self.label_resource_config = {
            'font': self.label_resource_font,
            'width': self.label_resource_width,
            'height': self.label_resource_height,
            'relief': 'solid',
            'borderwidth': self.label_resource_border_width,
            'bg': 'white',
        }

        self.label_user_image_config = {
            'bg': 'white',
            'highlightthickness': self.label_img_users_border_width,
        }

        self.label_user_text_config = {
            'font': self.label_user_font,
            'bg': 'white',
            'justify': LEFT,
        }

        self.frames_config = {
            'bg': 'white',
            # 'relief': 'solid',
            # 'borderwidth': 2,
        }

        self.labels_config = {
            'font': self.label_user_font,
            'bg': 'white',
        }

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
