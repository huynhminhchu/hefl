from enum import IntEnum, auto


class PacketType(IntEnum):
    def __eq__(self, o: int) -> bool:
        return self.value == o

    def _generate_next_value_(self, _start, count, _last_values):
        return count

    NONE = auto()
    SERVER_WELCOME_WORKER = auto()
    SERVER_WELCOME_LEADER = auto()
    CLIENT_HELLO = auto()
    INIT_MODEL = auto()
    REQUEST_HE_KEY = auto()
    REQUEST_MODEL = auto()
    REQUEST_AGGREGATE = auto()
    RESPONSE_HE_KEY = auto()
    RESPONSE_MODEL = auto()
    RESPONSE_AGGREGATE = auto()
