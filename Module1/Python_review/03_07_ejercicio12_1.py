import time

def execution_time():
    def wrapper():
        print(time.time())
    return wrapper

@execution_time

def sum_two_numbers(a: int, b: int) -> int:
    return (a + b)





