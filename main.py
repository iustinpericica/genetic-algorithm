from distutils.command.config import config
from types import SimpleNamespace
from math import ceil, log2
import random

INPUT_FILE_NAME = "input.txt"
OUTPUT_FILE_NAME = "output.txt"
SECTION_SEPARATOR = "\n" + 50 * "=" + "\n\n"

out_file = open(OUTPUT_FILE_NAME, "w")

def read_input(input_file_name):
    input_file = open(input_file_name, "r")

    config = SimpleNamespace()

    config.nr_of_population = int( input_file.readline() )

    config.function =  SimpleNamespace()

    config.function.domain =  SimpleNamespace()
    config.function.domain.left = int( input_file.readline() )
    config.function.domain.right = int( input_file.readline() )

    config.function.coefficients = SimpleNamespace()
    config.function.coefficients.a = int( input_file.readline() )
    config.function.coefficients.b = int( input_file.readline() )
    config.function.coefficients.c = int( input_file.readline() )

    config.genetics =  SimpleNamespace()
    config.genetics.precision = int(input_file.readline())
    config.genetics.prob_crossover = float(input_file.readline())
    config.genetics.prob_mutation = float(input_file.readline())

    config.nr_phases = int(input_file.readline())

    input_file.close()
    return config

def compute_chr_length(config):
    return ceil(log2((config.function.domain.right - config.function.domain.left) * pow(10, config.genetics.precision))) 

def compute_pop(config):
    chromosomes = [] 
    for _ in range(config.nr_of_population):
        chromosomes.append(compute_chromosome(config))
    return chromosomes

def compute_chromosome(config):
    chromosome = [] 
    for _ in range(config.chr_len):
        chromosome.append(round(random.random()))
    return chromosome

config = read_input(INPUT_FILE_NAME)
config.chr_len = compute_chr_length(config)
population = compute_pop(config)

print(population)

# for i in range(config.nr_of_population):
#     print_this_step = False
#     if i == 0:
#         print_this_step = True
    

# print(read_input(INPUT_FILE_NAME))