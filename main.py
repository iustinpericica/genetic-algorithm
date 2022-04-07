from copy import deepcopy
from distutils.command.config import config
from functools import reduce
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

def quadratic_fn(a, b, c, x):
  return a * pow(x, 2) + b * x + c

def get_chr_decimal(chr, left, right):
  decimal_value = int("".join(list(map(str, chr))), 2)

  return ((right - left) * decimal_value) / (pow(2, len(chr)) - 1) + left

def get_ch_values_f(config, population):
    coefficients = config.function.coefficients
    domain = config.function.domain
    return list(map(lambda ch: quadratic_fn(coefficients.a, coefficients.b, coefficients.c, get_chr_decimal(ch, domain.left, domain.right)), population))

def get_elitist_chr(config, population):
  ch_values = get_ch_values_f(config, population)
  
  (idx, max_fitness) = reduce(lambda acc, crt: (crt[0] if crt[1] == max(acc[1], crt[1]) else acc[0] , max(acc[1], crt[1])), zip(range(0, len(population)), ch_values))

  return (idx, population[idx], max_fitness)

def get_average_chr_only_fit(config, population):
  ch_values = get_ch_values_f(config, population)
  return (reduce(lambda acc, crt: acc + crt, ch_values)) / len(ch_values)

def get_chrs_probabilities(config, population):
    ch_values_f = get_ch_values_f(config, population)
    total_func_values = reduce(lambda acc, crt: acc + crt, ch_values_f)
    probabilities = []
    for i in range(0, len(population)):
        probabilities.append(ch_values_f[i] / total_func_values)
    return probabilities

def get_selection_intervals(config, population):
    probabilities = get_chrs_probabilities(config, population)
    last_interval_bound = 0
    selection_intervals = []
    for probability in probabilities:
        selection_intervals.append((last_interval_bound, last_interval_bound + probability))
        last_interval_bound += probability
    return selection_intervals

def search_probability_in_intervals_return_i_index(intervals, prob_chromosome):
  start, end = 0, len(intervals) - 1

  while start <= end:
    mid = (start + end) // 2

    (lb, ub) = intervals[mid]
    if lb <= prob_chromosome < ub:
      return mid
    
    if prob_chromosome > ub:
      start = mid + 1
    elif prob_chromosome < lb:
      end = mid - 1

def select_from_population(population, selection_intervals):
  selected_population = []
  
  for _ in population:
    prob_chromosome = random.uniform(0, 1)
    chromosome_idx = search_probability_in_intervals_return_i_index(selection_intervals, prob_chromosome)
    selected_population.append(deepcopy(population[chromosome_idx]))
  
  return selected_population

######## PRINT FUNCTIONS ##########

def print_population(config, population, out_file):
  if out_file == None:
    return
  
  coefficients = config.function.coefficients
  domain = config.function.domain

  for i in range(0, len(population)):
    chromosome = population[i]
    ch_value = get_chr_decimal(chromosome, domain.left, domain.right)
    func_value = quadratic_fn(coefficients.a, coefficients.b, coefficients.c, ch_value)

    out_file.write("\t")
    out_file.write("chromosome {}: {}; x = {}; f = {}".format(i + 1, "".join(list(map(str, chromosome))), ch_value, func_value))
    out_file.write("\n")

def get_crossover_and_not_crossover_population(population, config):
    in_crossover = []
    out_crossover = []
    
    for idx in range(0, len(population)):
        prob_chromosome = random.uniform(0, 1)
        chromosome = population[idx]

        if prob_chromosome >= config.genetics.prob_crossover:
            out_crossover.append(chromosome)
        else:
            in_crossover.append(chromosome)

    if in_crossover and len(in_crossover) % 2 == 1:
        out_crossover.append(in_crossover.pop())

    return (in_crossover, out_crossover)

def chromosomes_crossover(ch1, ch2, breakpoint):
  after_breakpoint_ch1 = ch1[breakpoint:]
  after_breakpoint_ch2 = ch2[breakpoint:]

  new_ch1 = ch1[:breakpoint] + after_breakpoint_ch2
  new_ch2 = ch2[:breakpoint] + after_breakpoint_ch1

  return (new_ch1, new_ch2)

def perform_crossover(population, config):
    result = []
    chr_length = config.chr_len
    for idx in range(0, len(population), 2):
      chr1 = population[idx]
      chr2 = population[idx + 1]
      breakpoint = random.randint(1, chr_length - 1)

      (new_ch1, new_ch2) = chromosomes_crossover(chr1, chr2, breakpoint)
      result.append(new_ch1)
      result.append(new_ch2)
    return result

def perform_mutation(population, config):
    ch_len = config.chr_len

    for idx in range(0, len(population)):
        chromosome = population[idx]
        old_chromosome = chromosome[:]

        has_mutation = False
        # 'current chromosome'

        for gene_idx in range(0, len(chromosome)):
            prob_gene = random.uniform(0, 1)
            if prob_gene >= config.genetics.prob_mutation:
                continue
            
            has_mutation = True

            old_gene = chromosome[gene_idx]
            chromosome[gene_idx] = 1 - chromosome[gene_idx]
    
    return population

#### START #####

out_file = open(OUTPUT_FILE_NAME, "w")
config = read_input(INPUT_FILE_NAME)
config.chr_len = compute_chr_length(config)
population = compute_pop(config)

for i in range(config.nr_of_population):
    first_step = False
    if i == 0:
        first_step = True
    
    elitist_chr = get_elitist_chr(config, population)
    probabilities_per_chr = get_chrs_probabilities(config, population)
    selection_intervals = get_selection_intervals(config, population)
    selected_population = select_from_population(population, selection_intervals)
    (pop_in_crossover, pop_out_crossover) = get_crossover_and_not_crossover_population(selected_population, config)
    pop_inc_after_cross = perform_crossover(pop_in_crossover, config)
    population = pop_inc_after_cross + pop_out_crossover # fresh population after cross
    population = perform_mutation(population, config)
    population.append(elitist_chr[1])

    if not first_step:
        out_file.write("\tthe max value is: {}; the average is: {}\n".format(elitist_chr[2], get_average_chr_only_fit(config, population)))
    
    if first_step:
        out_file.write("1. Initial population: \n")
        print_population(config, population, out_file)
        out_file.write(SECTION_SEPARATOR)

    

# print(read_input(INPUT_FILE_NAME))