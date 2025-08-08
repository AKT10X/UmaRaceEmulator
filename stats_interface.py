# stats_interface.py

class Skill:
    """
    スキル定義クラス。
    Attributes:
        name (str): スキル名
        activation_pos (float): 発動位置 [m]
        base_duration (float): 基礎持続時間 [s]
        effect (float): 効果量 (m/s または m/s²)
        effect_type (str): 'speed' または 'accel'
        rank_condition (int | None): 条件順位 (None は無条件)
    """
    def __init__(
        self,
        name: str,
        activation_pos: float,
        base_duration: float,
        effect: float,
        effect_type: str,
        rank_condition: int | None = None
    ):
        self.name = name
        self.activation_pos = activation_pos
        self.base_duration = base_duration
        self.effect = effect
        self.effect_type = effect_type  # 'speed' or 'accel'
        self.rank_condition = rank_condition

class UmaStats:
    """
    ウマ娘の各種ステータスと初期レーンを保持するクラス。
    Attributes:
        speed (float): 速度ステータス
        stamina (float): スタミナステータス
        power (float): パワーステータス
        guts (float): 根性ステータス
        wiz (float): 賢さステータス
        running_style (str): 脚質
        distance_apt (str): 距離適性
        initial_lane (float): シミュ開始レーン
        skills (List[Skill]): 保持スキルリスト
    """
    def __init__(
        self,
        speed: float,
        stamina: float,
        power: float,
        guts: float,
        wiz: float,
        running_style: str,
        distance_apt: str,
        initial_lane: float,
        skills: list
    ):
        self.speed = speed
        self.stamina = stamina
        self.power = power
        self.guts = guts
        self.wiz = wiz
        self.running_style = running_style
        self.distance_apt = distance_apt
        self.initial_lane = initial_lane
        self.skills = skills

class CharacterConfig:
    """
    シミュレーション用キャラ設定。
    Attributes:
        name (str): キャラ名
        stats (UmaStats): ステータス＆スキル情報
    """
    def __init__(self, name: str, stats: UmaStats):
        self.name = name
        self.stats = stats

# --- スキル例の定義 ---
# 目標速度スキル: "序盤巧者"
simple_speed_skill = Skill(
    name='序盤巧者',
    activation_pos=300.0,
    base_duration=1.8,
    effect=0.15,
    effect_type='speed',
    rank_condition=1
)
# 加速度スキル: "地固め"
accel_skill = Skill(
    name='地固め',
    activation_pos=0.0,
    base_duration=3.0,
    effect=0.15,
    effect_type='accel',
    rank_condition=None
)

# --- 各キャラクター設定例 ---
characters = [
    CharacterConfig(
        name='Special Week',
        stats=UmaStats(
            speed=1200.0,
            stamina=800.0,
            power=1200.0,
            guts=1000.0,
            wiz=900.0,
            running_style='逃げ',
            distance_apt='A',
            initial_lane=1.0,
            skills=[simple_speed_skill, accel_skill]
        )
    ),
    CharacterConfig(
        name='Silence Suzuka',
        stats=UmaStats(
            speed=1300.0,
            stamina=900.0,
            power=1100.0,
            guts=950.0,
            wiz=1000.0,
            running_style='先行',
            distance_apt='A',
            initial_lane=0.5,
            skills=[]
        )
    ),
    # 他キャラをここに追加
]


