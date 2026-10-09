"""CPU protocol tests; these do not execute a GPU transport or RL iteration."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from uccl_zip_credits import ConsumerCredits


class ConsumerCreditTest(unittest.TestCase):
    def test_slow_consumer_keeps_buffer_owned(self):
        credits = ConsumerCredits(2, {1, 2, 3, 4})
        credits.reserve(0, 0)
        for receiver in (3, 1, 4):
            self.assertFalse(credits.acknowledge(0, 0, receiver))
        with self.assertRaises(ValueError):
            credits.release(0, 0)
        with self.assertRaises(ValueError):
            credits.reserve(1, 0)
        self.assertTrue(credits.acknowledge(0, 0, 2))
        credits.release(0, 0)
        credits.reserve(1, 0)
        with self.assertRaises(ValueError):
            credits.acknowledge(0, 0, 2)
        self.assertEqual(credits.slots[0].remaining, {1, 2, 3, 4})

    def test_out_of_order_slots_do_not_release_each_other(self):
        credits = ConsumerCredits(16, {1, 2, 3, 4})
        for sequence in range(16):
            credits.reserve(sequence, sequence)
        for slot in reversed(range(16)):
            for receiver in (4, 2, 3, 1):
                complete = credits.acknowledge(slot, slot, receiver)
                self.assertEqual(complete, receiver == 1)
            credits.release(slot, slot)
        self.assertFalse(credits.slots)

    def test_duplicate_credit_cannot_replace_another_consumer(self):
        credits = ConsumerCredits(1, {1, 2})
        credits.reserve(0, 0)
        self.assertFalse(credits.acknowledge(0, 0, 1))
        for receiver in (1, 7):
            with self.assertRaises(ValueError):
                credits.acknowledge(0, 0, receiver)
        self.assertEqual(credits.slots[0].remaining, {2})

    def test_generation_reuse_remains_bounded(self):
        credits = ConsumerCredits(16, set(range(1, 5)))
        for sequence in range(8192):
            slot = sequence % 16
            credits.reserve(sequence, slot)
            for receiver in range(1, 5):
                credits.acknowledge(sequence, slot, receiver)
            credits.release(sequence, slot)
        with self.assertRaises(ValueError):
            credits.reserve(8191, 0)
        self.assertFalse(credits.slots)


if __name__ == "__main__":
    unittest.main()
