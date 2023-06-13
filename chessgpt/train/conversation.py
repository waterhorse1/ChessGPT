import dataclasses
from typing import List, Any

from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None

    def set_system(self, system_prompt):
        self.system = system_prompt

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "sep_style": self.sep_style,
            "sep": self.sep,
            "sep2": self.sep2,
        }


chess_agent_conversation = Conversation(
        system="A friendly, helpful chat between some humans.",
        roles=("Human 0", "Human 1"),
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="<|endoftext|>",
    )