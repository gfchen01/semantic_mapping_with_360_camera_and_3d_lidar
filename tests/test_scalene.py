import time

def my_function():
    total = 0
    for i in range(10000):
        total += i ** 2
        time.sleep(0.001)
    return total

if __name__ == '__main__':
    my_function()