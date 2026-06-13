import os
import logging
import logging.handlers


class ErrFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno == logging.ERROR


class InfoFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno == logging.INFO


class DebugFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno == logging.DEBUG


class WarnMoreFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno >= logging.WARNING


class InfoMoreFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno >= logging.INFO


class DebugMoreFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelno >= logging.DEBUG


class Filter(logging.Filter):
    def __init__(self, log_level: int):
        self.log_level = log_level
        super().__init__()

    def filter(self, record) -> bool:
        return record.levelno == self.log_level


class MoreFilter(Filter):
    def filter(self, record) -> bool:
        return record.levelno >= self.log_level


log_level_filter = {
    'err': ErrFilter,
    'info': InfoFilter,
    'debug': DebugFilter,
    'warn+': WarnMoreFilter,
    'info+': InfoMoreFilter,
    'debug+': DebugMoreFilter,
}


def init_logger(log_file_name):
    log = logging.getLogger(log_file_name.split(".")[0])
    log_fmt = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] [Thread-%(thread)d] %(filename)s:%(lineno)d - %(message)s')
    log_dir = "/data/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file_name)
    # 输出至文件
    # handler_file = logging.FileHandler(log_file_path)
    # 按天输出文件
    handler_file = logging.handlers.TimedRotatingFileHandler(log_file_path, when='midnight', interval=1,
                                                             backupCount=7, encoding='utf-8')
    handler_file.suffix = "%Y-%m-%d.log"
    handler_file.setFormatter(log_fmt)
    handler_file.addFilter(MoreFilter(logging.INFO))
    log.addHandler(handler_file)

    # 输出至控制台
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(log_fmt)
    handler_console.addFilter(MoreFilter(logging.INFO))
    log.addHandler(handler_console)

    # 全局日志等级
    log.setLevel(logging.INFO)
    return log




