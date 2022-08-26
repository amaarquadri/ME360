def q6():
    value = 0
    salary = 3000

    for year in range(40):
        for month in range(12):
            value += salary * 0.2
            value *= (1 + 0.06 / 12)
        salary *= (1 + 0.02)
    print(value)


if __name__ == '__main__':
    q6()
