import math


class Helper:
    @staticmethod
    def get_eps_threshold(e_start, e_end, e_decay, steps_done):
        return e_end + (e_start - e_end) * math.exp(-1. * steps_done / e_decay)


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)
