
class Job:
    def __init__(
            self,
            size_resource_slots: int,
    ) -> None:

        self.size_resource_slots: int = size_resource_slots
        self.priority: int = 0

    def set_priority(
            self,
            priority_level: int,
    ) -> None:

        self.priority = priority_level
