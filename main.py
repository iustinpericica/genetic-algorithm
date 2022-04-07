from types import SimpleNamespace


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

print(read_input(INPUT_FILE_NAME))