#!/usr/bin/env python

import math, random, re, warnings

import torch

######################################################################


class Task:
    def used_characters(self):
        raise NotImplementedError

    def generate_sample(self):
        raise NotImplementedError

    def correct(self, s):
        raise NotImplementedError


######################################################################


class TaskArithmeticQuizz(Task):

    def __init__(self, nb_numbers=4, valmax=12):
        self.nb_numbers = nb_numbers
        self.valmax = valmax

    def used_characters(self):
        return "_0123456789,+*/-=:()%"

    def generate_sample(self):
        def expression(numbers):
            l = len(numbers)
            if l == 1:
                return str(numbers[0])
            else:
                op = random.choice(["+", "-", "*", "/", "%"])
                k = random.randrange(1, l)
                return (
                    "(" + expression(numbers[:k]) + op + expression(numbers[k:]) + ")"
                )

        while True:
            try:
                numbers = [
                    v.item() for v in torch.randint(self.valmax, (self.nb_numbers,)) + 1
                ]
                e = expression(numbers)
                value = eval(e.replace("/", "//"))
                if value > 0:
                    random.shuffle(numbers)
                    return (
                        ",".join([str(n) for n in numbers]) + "=" + str(value) + ":" + e
                    )
            except Exception as e:
                # print(e)
                pass

    def correct(self, s):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            try:
                s = s.strip("_")
                prompt, answer = s.split(":")
                numbers, result = prompt.split("=")
                numbers = [int(n) for n in numbers.split(",")]
                numbers.sort()
                result = int(result)
                numbers_in_answer = re.sub("[^0-9][^0-9]*", ",", answer).strip(",")
                numbers_in_answer = [int(n) for n in numbers_in_answer.split(",")]
                numbers_in_answer.sort()
                result_in_answer = eval(answer.replace("/", "//"))
                return numbers_in_answer == numbers and result == result_in_answer
            except:
                return False


######################################################################
