"""Bounded consumer credits for reusable weight-transfer buffers."""

from dataclasses import dataclass


@dataclass
class Reservation:
    sequence: int
    remaining: set[int]


class ConsumerCredits:
    def __init__(self, depth: int, receivers: set[int]):
        if depth < 1 or not receivers:
            raise ValueError("Require positive depth and at least one receiver")
        self.depth = depth
        self.receivers = frozenset(receivers)
        self.slots: dict[int, Reservation] = {}
        self.last_sequence = -1

    def reserve(self, sequence: int, slot: int) -> None:
        if not 0 <= slot < self.depth or slot in self.slots:
            raise ValueError("Transfer slot is unavailable")
        if sequence <= self.last_sequence:
            raise ValueError("Transfer sequence must increase")
        self.slots[slot] = Reservation(sequence, set(self.receivers))
        self.last_sequence = sequence

    def acknowledge(self, sequence: int, slot: int, receiver: int) -> bool:
        reservation = self.slots.get(slot)
        if reservation is None or reservation.sequence != sequence:
            raise ValueError("Credit does not match the active slot generation")
        if receiver not in reservation.remaining:
            raise ValueError("Unknown receiver or duplicate consumer credit")
        reservation.remaining.remove(receiver)
        return not reservation.remaining

    def release(self, sequence: int, slot: int) -> None:
        reservation = self.slots.get(slot)
        if reservation is None or reservation.sequence != sequence or reservation.remaining:
            raise ValueError("Transfer still has outstanding consumers")
        del self.slots[slot]
