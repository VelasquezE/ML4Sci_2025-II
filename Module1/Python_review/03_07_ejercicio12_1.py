# Write a function that takes two numbers and returns their sum. 
#Include type hints and a docstring.
#Create a decorator that times how long a function takes to execute.
import time

def execution_time(func):
    """
    Decorator that measures and prints the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to be wrapped and timed.

    Returns
    -------
    callable
        The wrapped function. When called, it executes `func`, prints the
        elapsed time, and returns the result of `func`.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time
        print(f"{func.__name__ } executed in {total_time} seconds")
        return result 
    return wrapper

@execution_time

def sum_two_numbers(a: int, b: int) -> int:
    """
    Adds two numbers.

    Parameters:
    -----------
    a : int
    b : int

    Returns: 
    --------
    a + b
    """
    return (a + b)

print(f"La suma es igual a {sum_two_numbers(2, 2)}")





