from utils import random

from aenum import Enum, auto

from config import settings


class TimerType(Enum):
    COLLISION = auto()
    AGGREGATION = auto()
    SENSE = auto()


class TxTimer:
    def __init__(self, env, type):
        self.env = env
        self.timer = None
        self.type = type

        if type is TimerType.COLLISION:
            self.backoff = random(settings.TX_COLLISION_TIMER_RANDOM)
            self.backoff_min = settings.TX_COLLISION_TIMER_RANDOM[0]
            self.backoff_max = settings.TX_COLLISION_TIMER_RANDOM[1]
        elif type is TimerType.AGGREGATION:
            self.backoff = random(settings.TX_AGGREGATION_TIMER_RANDOM)
            self.backoff_max = settings.TX_AGGREGATION_TIMER_RANDOM[1]
            self.backoff_min = settings.TX_AGGREGATION_TIMER_RANDOM[0]
        else:
            self.backoff = settings.MEASURE_INTERVAL_S


    def reset(self):
        self.timer = None

    def step_up(self):
        """
        step up the running timer and adapt the backoff period
        :return:
        """
        if self.type is not TimerType.DATA:
            ValueError("Only Data timer can be stepped up/down")
        new_backoff = self.backoff + settings.TX_DATA_TIMER_STEP_UP
        if new_backoff < self.backoff_max:
            self.backoff = new_backoff
            self.inc_time_by(settings.TX_DATA_TIMER_STEP_UP)
        else:
            # we don't increase the timer
            self.backoff = self.backoff_max

    def step_down(self):
        if self.type is not TimerType.DATA:
            ValueError("Only Data timer can be stepped up/down")
        new_backoff = self.backoff - settings.TX_DATA_TIMER_STEP_DOWN
        self.backoff = self.backoff_min if new_backoff < self.backoff_min else new_backoff
        # no need to step down the actual running timer

    def start(self, restart=True):
        # alias for init backoff
        if restart:
            self.init_backoff()
        else:
            # only start when not yet running
            if self.timer is None:
                self.start()


    def init_backoff(self):
        self.set_timer(self.backoff)

    def set_timer(self, time, relative=True):
        """

        :param time:
        :param relative: interpret time as absolute or relative to current time
        :return:
        """
        offset = 0
        if relative:
            offset = self.env.now
        self.timer = offset + time

    def is_set(self):
        return self.timer is not None

    def is_expired(self):
        if self.timer is not None:
            return self.env.now > self.timer
        return False  # presume it is not expired as it is not set

    def inc_time_by(self, time):
        """
        Extend the timer by 'time' amount
        :param time:
        :return:
        """
        self.timer += time

