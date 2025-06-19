import time
from dataclasses import dataclass


@dataclass
class TimerLog:
    start_timestamp: float
    duration: float
    rep_num: int


class Timer:
    print_on = True

    def __init__(self, name=None) -> None:
        self.logs = {}
        self.active_channel = None
        self.name = name

    def log(self, channel):
        current_timestamp = time.time()

        if self.active_channel is not None:
            self.logs[self.active_channel].duration += current_timestamp - self.logs[self.active_channel].start_timestamp
            self.logs[self.active_channel].rep_num += 1
        self.active_channel = channel

        if channel is None:
            return

        if channel not in self.logs:
            self.logs[channel] = TimerLog(current_timestamp, 0, 0)
        else:
            self.logs[channel].start_timestamp = current_timestamp

    def stop(self):
        self.log(None)

    def total_duration(self):
        return sum([log.duration for log in self.logs.values()])

    def message(self):
        msg = ""

        descriptor = f" Timer Logs: {self.name} " if self.name else " Timer Logs "
        msg += "\n" + "=" * 15 + descriptor + "=" * 15 + "\n"
        seperator_len = 15 + len(descriptor) + 15

        for channel, log in self.logs.items():
            msg += f"{channel:<20}: {log.duration:>10.3f}s\n"

        msg += "-" * seperator_len + "\n"
        msg += f"{'total':<20}: {self.total_duration():>10.3f}s\n"
        msg += "=" * seperator_len + "\n"

        return msg

    def print(self):
        if not Timer.print_on:
            return
        print(self.message())
